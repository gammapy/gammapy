# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Image utility functions"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from multiprocessing import Pool
from functools import partial
import numpy as np
from astropy.coordinates import Angle
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits

__all__ = [
    'block_reduce_hdu',
    'image_groupby',
    'lon_lat_rectangle_mask',
    'lon_lat_circle_mask',
    'make_header',
    'process_image_pixels',
]

log = logging.getLogger(__name__)


def _fftconvolve_wrap(kernel, data):
    from scipy.signal import fftconvolve
    from scipy.ndimage.filters import gaussian_filter

    # wrap gaussian filter as a special case, because the gain in
    # performance is factor ~100
    if isinstance(kernel, Gaussian2DKernel):
        width = kernel.model.x_stddev.value
        norm = kernel.array.sum()
        return norm * gaussian_filter(data, width)
    else:
        return fftconvolve(data, kernel.array, mode='same')


def scale_cube(data, kernels, parallel=True):
    """
    Compute scale space cube.

    Compute scale space cube by convolving the data with a set of kernels and
    stack the resulting images along the third axis.

    Parameters
    ----------
    data : `~numpy.ndarray`
        Input data.
    kernels: list of `~astropy.convolution.Kernel`
        List of convolution kernels.
    parallel : bool
        Whether to use multiprocessing.

    Returns
    -------
    cube : `~numpy.ndarray`
        Array of the shape (len(kernels), data.shape)
    """
    wrap = partial(_fftconvolve_wrap, data=data)

    if parallel:
        pool = Pool()
        result = pool.map(wrap, kernels)
        pool.close()
        pool.join()
    else:
        result = map(wrap, kernels)
    return np.dstack(result)


def process_image_pixels(images, kernel, out, pixel_function):
    """Process images for a given kernel and per-pixel function.

    This is a helper function for the following common task:
    For a given set of same-shaped images and a smaller-shaped kernel,
    process each image pixel by moving the kernel at that position,
    cut out kernel-shaped parts from the images and call a function
    to compute output values for that position.

    This function loops over image pixels and takes care of bounding
    box computations, including image boundary handling.

    Parameters
    ----------
    images : dict of arrays
        Images needed to compute out
    kernel : array (shape must be odd-valued)
        kernel shape must be odd-valued
    out : single array or dict of arrays
        These arrays must have been pre-created by the caller
    pixel_function : function to process a part of the images

    Examples
    --------

    As an example, here is how to implement convolution as a special
    case of process_image_pixels with one input and output image::

        def convolve(image, kernel):
            '''Convolve image with kernel'''
            from gammapy.image.utils import process_image_pixels
            images = dict(image=np.asanyarray(image))
            kernel = np.asanyarray(kernel)
            out = dict(image=np.empty_like(image))
            def convolve_function(images, kernel):
                value = np.sum(images['image'] * kernel)
                return dict(image=value)
            process_image_pixels(images, kernel, out, convolve_function)
            return out['image']

    * TODO: add different options to treat the edges
    * TODO: implement multiprocessing version
    * TODO: this function is similar to view_as_windows in scikit-image:
            http://scikit-image.org/docs/dev/api/skimage.util.html#view-as-windows
            Is this function needed or can everything be done with view_as_windows?
    """
    if isinstance(out, dict):
        n0, n1 = out.values()[0].shape
    else:
        n0, n1 = out.shape

    # Check kernel shape
    k0, k1 = kernel.shape
    if (k0 % 2 == 0) or (k1 % 2 == 0):
        raise ValueError('Kernel shape must have odd dimensions')
    k0, k1 = (k0 - 1) / 2, (k1 - 1) / 2

    # Loop over all pixels
    for i0 in range(0, n0):
        for i1 in range(0, n1):
            # Compute low and high extension
            # (# pixels, not counting central pixel)
            i0_lo = min(k0, i0)
            i1_lo = min(k1, i1)
            i0_hi = min(k0, n0 - i0 - 1)
            i1_hi = min(k1, n1 - i1 - 1)

            # Cut out relevant parts of the image arrays
            # This creates views, i.e. is fast and memory efficient
            image_parts = dict()
            for name, image in images.items():
                # hi + 1 because with Python slicing the hi edge is not included
                part = image[i0 - i0_lo: i0 + i0_hi + 1,
                       i1 - i1_lo: i1 + i1_hi + 1]
                image_parts[name] = part

            # Cut out relevant part of the kernel array
            # This only applies when close to the edge
            # hi + 1 because with Python slicing the hi edge is not included
            kernel_part = kernel[k0 - i0_lo: k0 + i0_hi + 1,
                          k1 - i1_lo: k1 + i1_hi + 1]

            # Call pixel_function for this one part
            out_part = pixel_function(image_parts, kernel_part)

            if isinstance(out_part, dict):
                # Store output
                for name, image in out.items():
                    out[name][i0, i1] = out_part[name]
            else:
                out[i0, i1] = out_part


def image_groupby(images, labels):
    """Group pixel by labels.

    This function is similar to `scipy.ndimage.measurements.labeled_comprehension`,
    but more general because it supports multiple input and output images.

    Parameters
    ----------
    images : list of `~numpy.ndarray`
        List of image objects.
    labels : `~numpy.ndarray`
        Labels for pixel grouping.

    Returns
    -------
    groups : list of `~numpy.ndarray`
        Grouped pixels acording to the labels.
    """
    for image in images:
        assert image.shape == labels.shape

    # Store data in 1D data frame (i.e. as pixel lists)
    # TODO: should we use array.flat or array.ravel() here?
    # It's not clear to me what the difference is and which is more efficient here.
    data = dict()
    data['labels'] = labels.flat
    for name, values in images.items():
        data[name] = values.flat

    # Group pixels by labels
    groups = data.groupby('labels')

    return groups
    # out = groups.aggregate(function)
    # return out


def make_header(nxpix=100, nypix=100, binsz=0.1, xref=0, yref=0,
                proj='CAR', coordsys='GAL',
                xrefpix=None, yrefpix=None):
    """Generate a FITS header from scratch.

    Uses the same parameter names as the Fermi tool gtbin.

    If no reference pixel position is given it is assumed ot be
    at the center of the image.

    Parameters
    ----------
    nxpix : int, optional
        Number of pixels in x axis. Default is 100.
    nypix : int, optional
        Number of pixels in y axis. Default is 100.
    binsz : float, optional
        Bin size for x and y axes in units of degrees. Default is 0.1.
    xref : float, optional
        Coordinate system value at reference pixel for x axis. Default is 0.
    yref : float, optional
        Coordinate system value at reference pixel for y axis. Default is 0.
    proj : string, optional
        Projection type. Default is 'CAR' (cartesian).
    coordsys : {'CEL', 'GAL'}, optional
        Coordinate system. Default is 'GAL' (Galactic).
    xrefpix : float, optional
        Coordinate system reference pixel for x axis. Default is None.
    yrefpix: float, optional
        Coordinate system reference pixel for y axis. Default is None.

    Returns
    -------
    header : `~astropy.io.fits.Header`
        Header
    """
    nxpix = int(nxpix)
    nypix = int(nypix)
    if not xrefpix:
        xrefpix = (nxpix + 1) / 2.
    if not yrefpix:
        yrefpix = (nypix + 1) / 2.

    if coordsys == 'CEL':
        ctype1, ctype2 = 'RA---', 'DEC--'
    elif coordsys == 'GAL':
        ctype1, ctype2 = 'GLON-', 'GLAT-'
    else:
        raise Exception('Unsupported coordsys: {}'.format(proj))

    pars = {'NAXIS': 2, 'NAXIS1': nxpix, 'NAXIS2': nypix,
            'CTYPE1': ctype1 + proj,
            'CRVAL1': xref, 'CRPIX1': xrefpix, 'CUNIT1': 'deg', 'CDELT1': -binsz,
            'CTYPE2': ctype2 + proj,
            'CRVAL2': yref, 'CRPIX2': yrefpix, 'CUNIT2': 'deg', 'CDELT2': binsz,
            }

    header = fits.Header()
    header.update(pars)

    return header


def block_reduce_hdu(input_hdu, block_size, func, cval=0):
    """Provides block reduce functionality for image HDUs.

    See http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce

    Parameters
    ----------
    image_hdu : `~astropy.io.fits.ImageHDU`
        Original image HDU, unscaled
    block_size : `~numpy.ndarray`
        Array containing down-sampling integer factor along each axis.
    func : callable
        Function object which is used to calculate the return value for each local block.
        This function must implement an axis parameter such as `numpy.sum` or `numpy.mean`.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the block size. Default 0.

    Returns
    -------
    image_hdu : `~astropy.io.fits.ImageHDU`
        Rebinned Image HDU
    """
    from skimage.measure import block_reduce

    header = input_hdu.header.copy()
    data = input_hdu.data
    # Define new header values for new resolution
    header['CDELT1'] = header['CDELT1'] * block_size[0]
    header['CDELT2'] = header['CDELT2'] * block_size[1]
    header['CRPIX1'] = ((header['CRPIX1'] - 0.5) / block_size[0]) + 0.5
    header['CRPIX2'] = ((header['CRPIX2'] - 0.5) / block_size[1]) + 0.5
    if len(input_hdu.data.shape) == 3:
        block_size = (1, block_size[1], block_size[0])
    elif len(input_hdu.data.shape) == 2:
        block_size = (block_size[1], block_size[0])
    data_reduced = block_reduce(data, block_size, func, cval)
    # Put rebinned data into a fitsHDU
    rebinned_image = fits.ImageHDU(data=data_reduced, header=header)
    return rebinned_image


def lon_lat_rectangle_mask(lons, lats, lon_min=None, lon_max=None,
                           lat_min=None, lat_max=None):
    """Produces a rectangular boolean mask array based on lat and lon limits.

    Parameters
    ----------
    lons : `~numpy.ndarray`
        Array of longitude values.
    lats : `~numpy.ndarray`
        Array of latitude values.
    lon_min : float, optional
        Minimum longitude of rectangular mask.
    lon_max : float, optional
        Maximum longitude of rectangular mask.
    lat_min : float, optional
        Minimum latitude of rectangular mask.
    lat_max : float, optional
        Maximum latitude of rectangular mask.

    Returns
    -------
    mask : `~numpy.ndarray`
        Boolean mask array for a rectangular sub-region defined by specified
        maxima and minima lon and lat.
    """
    if lon_min is not None:
        mask_lon_min = (lon_min <= lons)
    else:
        mask_lon_min = np.ones(lons.shape, dtype=bool)

    if lon_max is not None:
        mask_lon_max = (lons < lon_max)
    else:
        mask_lon_max = np.ones(lons.shape, dtype=bool)

    lon_mask = mask_lon_min & mask_lon_max

    if lat_min is not None:
        mask_lat_min = (lat_min <= lats)
    else:
        mask_lat_min = np.ones(lats.shape, dtype=bool)

    if lat_max is not None:
        mask_lat_max = (lats < lat_max)
    else:
        mask_lat_max = np.ones(lats.shape, dtype=bool)

    lat_mask = mask_lat_min & mask_lat_max

    return lon_mask & lat_mask


def lon_lat_circle_mask(lons, lats, center_lon, center_lat, radius):
    """Produces a circular boolean mask array.

    Parameters
    ----------
    lons : `~astropy.coordinates.Longitude`
        Array of longitude values.
    lats : `~astropy.coordinates.Latitude`
        Array of latitude values.
    center_lon : `~astropy.coordinates.Longitude`
        Longitude of center of circular mask.
    center_lat : `~astropy.coordinates.Latitude`
        Latitude of center of circular mask.
    radius : `~astropy.coordinates.Angle`
        Radius of circular mask.

    Returns
    -------
    mask : `~numpy.ndarray`
        Boolean mask array for a circular sub-region
    """
    lons.wrap_angle = Angle('180 deg')
    center_lon.wrap_angle = Angle('180 deg')

    mask = (lons - center_lon) ** 2 + (lats - center_lat) ** 2 < radius ** 2
    return mask

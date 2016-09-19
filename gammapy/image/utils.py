# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Image utility functions"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import WCS
from ..utils.wcs import get_wcs_ctype
from ..utils.energy import EnergyBounds
# TODO:
# Remove this when/if https://github.com/astropy/astropy/issues/4429 is fixed
from astropy.utils.exceptions import AstropyDeprecationWarning

__all__ = [
    'bin_events_in_image',
    'binary_disk',
    'binary_ring',
    'block_reduce_hdu',
    'disk_correlate',
    'image_groupby',
    'lon_lat_rectangle_mask',
    'lon_lat_circle_mask',
    'make_header',
    'process_image_pixels',
    'ring_correlate',
    'wcs_histogram2d',
]

log = logging.getLogger(__name__)


def _get_structure_indices(radius):
    """Get arrays of indices for a symmetric structure.

    Always generate an odd number of pixels and 0 at the center.

    Parameters
    ----------
    radius : float
        Structure radius in pixels.

    Returns
    -------
    y, x : mesh-grid `~numpy.ndarrays` all of the same dimensions
        Structure indices arrays.
    """
    radius = int(radius)
    y, x = np.mgrid[-radius: radius + 1, -radius: radius + 1]
    return x, y


def binary_disk(radius):
    """Generate a binary disk mask.

    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.

    Parameters
    ----------
    radius : float
        Disk radius in pixels

    Returns
    -------
    structure : `numpy.array`
        Structure element (bool array)
    """
    x, y = _get_structure_indices(radius)
    structure = x ** 2 + y ** 2 <= radius ** 2
    return structure


def binary_ring(r_in, r_out):
    """Generate a binary ring mask.

    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.

    Parameters
    ----------
    r_in : float
        Ring inner radius in pixels
    r_out : float
        Ring outer radius in pixels

    Returns
    -------
    structure : `numpy.array`
        Structure element (bool array)
    """
    x, y = _get_structure_indices(r_out)
    mask1 = r_in ** 2 <= x ** 2 + y ** 2
    mask2 = x ** 2 + y ** 2 <= r_out ** 2
    return mask1 & mask2


def disk_correlate(image, radius, mode='constant'):
    """Correlate image with binary disk kernel.

    Parameters
    ----------
    image : `~numpy.ndarray`
        Image to be correlated.
    radius : float
        Disk radius in pixels.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        the mode parameter determines how the array borders are handled.
        For 'constant' mode, values beyond borders are set to be cval.
        Default is 'constant'.

    Returns
    -------
    convolve : `~numpy.ndarray`
        The result of convolution of image with disk of given radius.

    """
    from scipy.ndimage import convolve
    structure = binary_disk(radius)
    return convolve(image, structure, mode=mode)


def ring_correlate(image, r_in, r_out, mode='constant'):
    """Correlate image with binary ring kernel.

    Parameters
    ----------
    image : `~numpy.ndarray`
        Image to be correlated.
    r_in : float
        Ring inner radius in pixels.
    r_out : float
        Ring outer radius in pixels.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        the mode parameter determines how the array borders are handled.
        For 'constant' mode, values beyond borders are set to be cval.
        Default is 'constant'.

    Returns
    -------
    convolve : `~numpy.ndarray`
        The result of convolution of image with ring of given inner and outer radii.
    """
    from scipy.ndimage import convolve
    structure = binary_ring(r_in, r_out)
    return convolve(image, structure, mode=mode)


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


def wcs_histogram2d(header, lon, lat, weights=None):
    """Histogram in world coordinates.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        FITS Header
    lon, lat : `~numpy.ndarray`
        World coordinates
    weights : `~numpy.ndarray`, optional
        Weights

    Returns
    -------
    histogram : `~astropy.io.fits.ImageHDU`
        Histogram

    See also
    --------
    numpy.histogramdd
    """
    if weights is None:
        weights = np.ones_like(lon)

    # Get pixel coordinates
    wcs = WCS(header)
    origin = 0  # convention for gammapy
    xx, yy = wcs.wcs_world2pix(lon, lat, origin)
    # Histogram pixel coordinates with appropriate binning.
    # This was checked against the `ctskymap` ctool
    # http://cta.irap.omp.eu/ctools/
    shape = header['NAXIS2'], header['NAXIS1']
    bins = np.arange(shape[0] + 1) - 0.5, np.arange(shape[1] + 1) - 0.5
    data = np.histogramdd([yy, xx], bins, weights=weights)[0]

    # return fits.ImageHDU(data, header, name='COUNTS')
    return fits.PrimaryHDU(data, header)


def bin_events_in_image(events, reference_image):
    """Bin events into an image.

    Parameters
    ----------
    events : `~gammapy.events.data.EventList`
        Event list table
    reference_image : `~astropy.io.fits.ImageHDU`
        An image defining the spatial bins.

    Returns
    -------
    count_image : `~astropy.io.fits.ImageHDU`
        Count image
    """
    if 'GLON' in reference_image.header['CTYPE1']:
        pos = events.galactic
    else:
        pos = events.radec

    return wcs_histogram2d(reference_image.header, pos.data.lon.deg, pos.data.lat.deg)


def _bin_events_in_cube(events, wcs, shape, energies=None, origin=0):
    """Bin events in LON-LAT-Energy cube.
    Parameters
    ----------
    events : `~astropy.data.EventList`
        Event list table
    wcs : `~astropy.wcs.WCS`
        WCS instance defining celestial coordinates.
    shape : tuple
        Tuple defining the spatial shape.
    energies : `~gammapy.utils.energy.EnergyBounds`
        Energy bounds defining the binning. If None only one energy bin defined
        by the minimum and maximum event energy is used.
    origin : {0, 1}
        Pixel coordinate origin.

    Returns
    -------
    data : `~numpy.ndarray`
        Counts cube.
    """
    if get_wcs_ctype(wcs) == 'galactic':
        galactic = events.galactic
        lon, lat = galactic.l.deg, galactic.b.deg
    else:
        lon, lat = events['RA'], events['DEC']

    xx, yy = wcs.wcs_world2pix(lon, lat, origin)
    event_energies = events['ENERGY']

    if energies is None:
        emin = np.min(event_energies)
        emax = np.max(event_energies)
        energies = EnergyBounds.equal_log_spacing(emin, emax, nbins=1, unit='TeV')
        shape = (2,) + shape

    zz = np.searchsorted(energies.value, event_energies.data)
    # Histogram pixel coordinates with appropriate binning.
    # This was checked against the `ctskymap` ctool
    # http://cta.irap.omp.eu/ctools/
    bins = np.arange(shape[0]), np.arange(shape[1] + 1) - 0.5, np.arange(shape[2] + 1) - 0.5
    return Quantity(np.histogramdd([zz, yy, xx], bins)[0], 'count')


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
        raise Exception('Unsupported coordsys: {0}'.format(proj))

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
    if lon_min:
        mask_lon_min = (lon_min <= lons)
    else:
        mask_lon_min = np.ones(lons.shape, dtype=bool)
    if lon_max:
        mask_lon_max = (lons < lon_max)
    else:
        mask_lon_max = np.ones(lons.shape, dtype=bool)

    lon_mask = mask_lon_min & mask_lon_max

    if lat_min:
        mask_lat_min = (lat_min <= lats)
    else:
        mask_lat_min = np.ones(lats.shape, dtype=bool)
    if lon_max:
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

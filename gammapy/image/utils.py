# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Image utility functions"""
from __future__ import division
import numpy as np

__all__ = ['tophat_correlate', 'ring_correlate', 'lookup', 'exclusion_distance',
           'atrous_image', 'atrous_hdu', 'process_image_pixels']


def _get_structure_indices(radius):
    """Get arrays of indices for a symmetric structure,
    i.e. with an odd number of pixels and 0 at the center.
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
    structure : array
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
    structure : array
        Structure element (bool array)
    """
    x, y = _get_structure_indices(r_out)
    mask1 = r_in ** 2 <= x ** 2 + y ** 2
    mask2 = x ** 2 + y ** 2 <= r_out ** 2
    return mask1 & mask2


def tophat_correlate(image, radius, mode='constant'):
    """Correlate image with binary disk kernel.
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    from scipy.ndimage import convolve
    structure = binary_disk(radius)
    return convolve(image, structure, mode=mode)


def ring_correlate(image, r_in, r_out, mode='constant'):
    """Correlate image with binary ring kernel.
    """
    from scipy.ndimage import convolve
    structure = binary_ring(r_in, r_out)
    return convolve(image, structure, mode=mode)


def exclusion_distance(exclusion):
    """Compute distance map, i.e. the Euclidean (=Cartesian 2D)
    distance (in pixels) to the nearest exclusion region.

    We need to call distance_transform_edt twice because it only computes
    dist for pixels outside exclusion regions, so to get the
    distances for pixels inside we call it on the inverted mask
    and then combine both distance images into one, using negative
    distances (note the minus sign) for pixels inside exclusion regions.
    """
    from scipy.ndimage import distance_transform_edt
    exclusion = np.asanyarray(exclusion, bool)
    distance_outside = distance_transform_edt(exclusion)
    distance_inside = distance_transform_edt(np.invert(exclusion))
    distance = np.where(exclusion, distance_outside, -distance_inside)
    return distance


def _lookup_pix(image, x, y):
    """
    image = numpy array
    x, y = array_like of pixel coordinates (floats OK)
    """
    # Find the right pixel
    x_int = np.round(x).astype(int)
    y_int = np.round(y).astype(int)

    # Return it's value
    # Note that numpy has index order (y, x)
    values = image[y_int, x_int]
    return values


def _lookup_world(image, lon, lat):
    """Look up values in an image
    image = astropy.io.fits.HDU
    lon, lat = world coordinates (float OK)
    """
    from astropy.wcs import WCS
    wcs = WCS(image.header)
    x, y = wcs.wcs_world2pix(lon, lat, 0)
    return _lookup_pix(image.data, x, y)


def lookup(image, x, y, world=True):
    """Look up values in an image

    TODO: document
    """
    if world:
        return _lookup_world(image, x, y)
    else:
        return _lookup_pix(image, x, y)

"""Compute common kernels for TS maps

A kernel is a source excess images after PSF convolution.
TODO: integrate over bins to get accurate kernels.
"""


'''
class KernelCalculator(object):
    """Compute PSF-convolved source images,
    to be used as kernels in the TS calculation"""

    def __init__(self, size=10, source='gauss', psf='gauss'):
        self.size = size

    def compute
'''


def atrous_image(image, n_levels):
    """Compute a trous transform for a given image.

    Parameters
    ----------
    image : 2D array
        Input image
    
    n_levels : integer
        Number of wavelet scales.
    
    Returns
    -------
    images : list of 2D arrays
        Wavelet transformed images.
    """
    # https://code.google.com/p/image-funcut/
    from imfun import atrous
    return atrous.decompose2d(image, level=n_levels)


def atrous_hdu(hdu, n_levels):
    """Compute a trous transform for a given FITS HDU.

    Parameters
    ----------
    hdu : 2D image HDU
        Input image

    n_levels : integer
        Number of wavelet scales.

    Returns
    -------
    images : HDUList
        Wavelet transformed images.
    """
    import logging
    from astropy.io import fits
    image = hdu.data
    logging.info('Computing a trous transform for {0} levels ...'.format(n_levels))
    images = atrous_image(image, n_levels)
    hdus = fits.HDUList()

    for level, image in enumerate(images):
        if level < len(images) - 1:
            name = 'level_{0}'.format(level)
        else:
            name = 'residual'
        scale_pix = 2 ** level
        scale_deg = hdu.header['CDELT2'] * scale_pix
        logging.info('HDU name = {0:10s}: scale = {1:5d} pix = {2:10.5f} deg'
                     ''.format(name, scale_pix, scale_deg))
        hdus.append(fits.ImageHDU(data=image, header=hdu.header, name=name))

    return hdus

def coordinates(image, world=True, lon_sym=True, radians=False):
    """Get coordinate images for a given image.

    This function is useful if you want to compute
    an image with values that are a function of position.

    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
    world : bool
        Use world coordinates (or pixel coordinates)?
    lon_sym : bool
        Use symmetric longitude range `(-180, 180)` (or `(0, 360)`)?

    Returns
    -------
    (lon, lat) : tuple of arrays
        Images as numpy arrays with values
        containing the position of the given pixel.

    Examples
    --------
    >>> l, b = coordinates(image)
    >>> dist = sqrt( (l-42)**2 + (b-43)**2)
    """
    # Create arrays of pixel coordinates
    y, x = np.indices(image.data.shape, dtype='int32') + 1

    if not world:
        return x, y

    from astropy.wcs import WCS
    wcs = WCS(image.header)
    lon, lat = wcs.wcs_pix2world(x, y, 1)
    
    if lon_sym:
        lon = np.where(lon > 180, lon - 360, lon)
    
    if radians:
        lon = np.radians(lon)
        lat = np.radians(lat)

    return lon, lat


def separation(image, center, world=True, radians=False):
    """Compute distance image from a given center point.
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    x_center, y_center = center
    x, y = coordinates(image, world=world, radians=radians)
    
    if world:
        from ..utils.coordinates import separation as sky_dist
        d = sky_dist(x, y, x_center, y_center)
        if radians:
            d = np.radians(d)
    else:
        d = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    return d

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
    
    Similar to scipy.ndimage.labeled_comprehension, only that here multiple inputs
    and outputs are supported.
    https://github.com/scipy/scipy/blob/master/scipy/ndimage/measurements.py#L270
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
    #out = groups.aggregate(function)
    #return out

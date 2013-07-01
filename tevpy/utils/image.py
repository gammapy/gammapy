"""Image utility functions"""
import numpy as np

__all__ = ['tophat_correlate', 'ring_correlate', 'lookup', 'exclusion_distance',
           'atrous_image', 'atrous_hdu']


def _get_structure_indices(radius):
    """
    Get arrays of indices for a symmetric structure,
    i.e. with an odd number of pixels and 0 at the center
    """
    radius = int(radius)
    y, x = np.mgrid[-radius: radius + 1, -radius: radius + 1]
    return x, y


def binary_disk(radius):
    """
    Generate a binary disk.
    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.
    """
    x, y = _get_structure_indices(radius)
    structure = x ** 2 + y ** 2 <= radius ** 2
    return structure


def binary_ring(r_in, r_out):
    """
    Generate a binary ring.
    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.
    """
    x, y = _get_structure_indices(r_out)
    mask1 = r_in ** 2 <= x ** 2 + y ** 2
    mask2 = x ** 2 + y ** 2 <= r_out ** 2
    return mask1 & mask2


def tophat_correlate(data, radius, mode='constant'):
    """
    Correlate with disk of given radius
    """
    from scipy.ndimage import convolve
    structure = binary_disk(radius)
    return convolve(data, structure, mode=mode)


def ring_correlate(data, r_in, r_out, mode='constant'):
    """
    Correlate with ring of given radii
    """
    from scipy.ndimage import convolve
    structure = binary_ring(r_in, r_out)
    return convolve(data, structure, mode=mode)


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

    image : 2d array
    n_levels : integer
    returns : list of 2d arrays
    """
    # https://code.google.com/p/image-funcut/
    from imfun import atrous
    return atrous.decompose2d(image, level=n_levels)


def atrous_hdu(hdu, n_levels):
    """Compute a trous transform for a given FITS HDU

    hdu : 2d image HDU
    n_levels : integer
    returns : HDUList
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
    """
    Get coordinate images for a given image.

    TODO: clean up (numpy docstring, add "use_" to kwargs)

    image: astropy.io.fits.ImageHDU
    world: use world coordinates or pixel coordinates?
    lon_sym: use longitude range (-180, 180) or (0, 360)

    Returns tuple (lon, lat) of maps as numpy arrays with values
    containing the position of the given pixel.

    This function is useful if you want to compute
    an image with values that are a function of position.

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

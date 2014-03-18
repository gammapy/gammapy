# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Measure image properties.
"""
from __future__ import print_function, division
import numpy as np

__all__ = ['BoundingBox',
           'find_max',
           'lookup',
           'lookup_max',
           'measure_containment_fraction',
           'measure_containment_radius',
           'measure_image_moments',
           'measure_labeled_regions',
           ]


class BoundingBox(object):
    """Rectangular bounding box.
    
    Parameters
    ----------
    TODO
    """
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax


    @staticmethod
    def for_circle(x, y, radius, nx, ny):
        """Compute bounding box for a circle.
        
        Parameters
        ----------
        TODO
        
        Returns
        -------
        TODO
        """
        x, y, radius = int(x), int(y), int(radius)
        xmin = max(x - radius, 0)
        xmax = min(x + radius, nx)
        ymin = max(y - radius, 0)
        ymax = min(y + radius, ny)
        bbox = BoundingBox(xmin, xmax, ymin, ymax)

        return bbox

        
    def to_ftcopy_string(self):
        """Create bounding box string in ftcopy format.
        
        Examples
        --------
        TODO
        """ 
        return '[{xmin}:{xmax},{ymin}:{ymax}]'.format(**self)


def measure_labeled_regions(data, labels, tag='IMAGE',
                            measure_positions=True, measure_values=True,
                            fits_offset=True, bbox_offset=True):
    """Measure source properties in image.
    
    Sources are defined by a label image.
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    import scipy.ndimage as nd
    from astropy.table import Table, Column
    # Measure all segments
    nsegments = labels.max()
    index = np.arange(1, nsegments + 1)  # Measure all sources
    # Measure stuff
    sum = nd.sum(data, labels, index)
    max = nd.maximum(data, labels, index)
    mean = nd.mean(data, labels, index)
    x, y = _split_xys(nd.center_of_mass(data, labels, index))
    xpeak, ypeak = _split_xys(nd.maximum_position(data, labels, index))
    xmin, xmax, ymin, ymax = _split_slices(nd.find_objects(labels))
    area = _measure_area(labels)
    # Use FITS convention, i.e. start counting at 1
    FITS_OFFSET = 1 if fits_offset else 0
    # Use SExtractor convention, i.e. slice max is inside
    BBOX_OFFSET = -1 if bbox_offset else 0
    # Create a table
    table = Table()
    table.add_column(Column(data=index, name='NUMBER'))

    if measure_positions:
        table.add_column(Column(data=x + FITS_OFFSET, name='X_IMAGE'))
        table.add_column(Column(data=y + FITS_OFFSET, name='Y_IMAGE'))
        table.add_column(Column(data=xpeak + FITS_OFFSET, name='XPEAK_IMAGE'))
        table.add_column(Column(data=ypeak + FITS_OFFSET, name='YPEAK_IMAGE'))
        table.add_column(Column(data=xmin + FITS_OFFSET, name='XMIN_IMAGE'))
        table.add_column(Column(data=xmax + FITS_OFFSET + BBOX_OFFSET, name='XMAX_IMAGE'))
        table.add_column(Column(data=ymin + FITS_OFFSET, name='YMIN_IMAGE'))
        table.add_column(Column(data=ymax + FITS_OFFSET + BBOX_OFFSET, name='YMAX_IMAGE'))
        table.add_column(Column(data=area, name='AREA'))

    if measure_values:
        table.add_column(Column(data=max, name=tag + '_MAX'))
        table.add_column(Column(data=sum, name=tag + '_SUM'))
        table.add_column(Column(data=mean, name=tag + '_MEAN'))

    return table


def find_max(image):
    """Find position of maximum in an image.
    
    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
        Input image
    
    Returns
    -------
    lon, lat, value : float
        Maximum value and its position 
    """
    from scipy.ndimage import maximum_position
    from astropy.wcs import WCS
    proj = WCS(image.header)
    data = image.data
    data[np.isnan(data)] = -np.inf
    y, x = maximum_position(data)
    GLON, GLAT = proj.wcs_pix2world(x, y, 0)
    val = data[int(y), int(x)]
    return GLON, GLAT, val


def _lookup_pix(image, x, y):
    """Look up values in an image for given pixel coordinates.
    
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
    """Look up values in an image for given world coordinates.
    
    image = astropy.io.fits.HDU
    lon, lat = world coordinates (float OK)
    """
    from astropy.wcs import WCS
    wcs = WCS(image.header)
    x, y = wcs.wcs_world2pix(lon, lat, 0)
    return _lookup_pix(image.data, x, y)


def lookup(image, x, y, world=True):
    """Look up values in an image.

    Parameters
    ----------
    image : array_like if world=False, astropy.io.fits.ImageHDU if world=True
        Array or image to look up the value
    x : array_like
        Array of X lookup positions
    y : array_like
        Array of Y lookup positions
    world : bool
        Are (x, y) WCS coordinates?
    """
    if world:
        return _lookup_world(image, x, y)
    else:
        return _lookup_pix(image, x, y)


def lookup_max(image, GLON, GLAT, theta):
    """Look up the max image values within a circle of radius theta
    around lists of given positions (nan if outside)"""
    from .utils import coordinates
    GLON = np.asarray(GLON)
    GLON = np.where(GLON > 180, GLON - 360, GLON)
    GLAT = np.asarray(GLAT)
    n_pos = len(GLON)
    theta = np.asarray(theta) * np.ones(n_pos, dtype='float32')

    ll, bb = coordinates(image)

    val = np.nan * np.ones(n_pos, dtype='float32')
    for ii in range(n_pos):
        mask = ((GLON[ii] - ll) ** 2 + 
                (GLAT[ii] - bb) ** 2 <= 
                theta[ii] ** 2)
        try:
            val[ii] = image.dat[mask].max()
        except ValueError:
            pass
    return val


def measure_image_moments(image, shift=0.5):
    """Compute 0th, 1st and 2nd moments of an image.

    NaN values are ignored in the computation.

    Parameters
    ----------
    image : array
        Input image array.
    shift : float (default value 0.5)
        Depending on where the image values are given, the grid has to be
        shifted. If the values are given at the center of the pixel
        shift = 0.5.

    Returns
    -------
    image moments : list
        List of image moments:
        [A, x_cms, y_cms, x_sigma, y_sigma, sqrt(x_sigma * y_sigma)]
        All value are given in pixel coordinates.
    """
    A = image[np.isfinite(image)].sum()
    y, x = np.indices(image.shape) + shift

    # Center of mass
    x_cms = (x * image)[np.isfinite(image)].sum() / A
    y_cms = (y * image)[np.isfinite(image)].sum() / A

    # Second moments
    x_var = ((x - x_cms) ** 2 * image)[np.isfinite(image)].sum() / A
    y_var = ((y - y_cms) ** 2 * image)[np.isfinite(image)].sum() / A
    x_sigma = np.sqrt(x_var)
    y_sigma = np.sqrt(y_var)

    return A, x_cms, y_cms, x_sigma, y_sigma, np.sqrt(x_sigma * y_sigma)


def measure_containment_radius(x_pos, y_pos, image, containment_fraction,
                               shift=0.5, normalize_image=True):
    """Measure containment radius.

    Uses `scipy.optimize.brentq`.

    Parameters
    ----------
    x_pos : float
        x position of the source in pixel coordinates.
    y_pos : float
        y position of the source in pixel coordinates.
    image : array
        Intensity image
    containment_fraction : float
        Containment fraction
    normalize_image : bool
        Set ``normalize_image=False`` if it already is normalized
        and you want the speed.

    Returns
    -------
    containment_radius : float
        Containment radius (pix)
    """
    from scipy.optimize import brentq

    # Set up squared radius array
    y, x = np.indices(image.shape) + shift
    rr = (x - x_pos) ** 2 + (y - y_pos) ** 2

    if normalize_image:
        image = image / image[np.isfinite(image)].sum()

    def func(r):
        return measure_containment_fraction(r, rr, image) - containment_fraction

    containment_radius = brentq(func, a=0, b=np.sqrt(rr.max()))
    
    return containment_radius


def measure_containment_fraction(r, rr, image):
    """Measure containment fraction.

    Parameters
    ----------
    r : float
        Containment radius.
    rr : array
        Squared radius array.
    image : array
        The image has to be normalized! I.e. image.sum() = 1.


    Returns
    -------
    containment_fraction : float
        Containment fraction
    """
    # Set up indices and containment mask
    containment_mask = rr < r ** 2
    mask = np.logical_and(np.isfinite(image), containment_mask)

    containment_fraction = image[mask].sum()

    return containment_fraction


def _split_xys(pos):
    """Helper function to work with `scipy.ndimage`."""
    x = np.array(pos)[:, 1]
    y = np.array(pos)[:, 0]
    return x, y


def _split_slices(slices):
    """Helper function to work with `scipy.ndimage`."""
    # scipy.ndimage.find_objects returns a list of
    # tuples of slices, which is not what we want.
    # The following list comprehensions extract
    # the format we need.
    xmin = np.asarray([t[1].start for t in slices])
    xmax = np.asarray([t[1].stop for t in slices])
    ymin = np.asarray([t[0].start for t in slices])
    ymax = np.asarray([t[0].stop for t in slices])
    return xmin, xmax, ymin, ymax


def _measure_area(labels):
    """Measure the area in pix of each segment."""
    nsegments = labels.max()
    area = np.zeros(nsegments)
    for i in range(nsegments):
        area[i] = (labels == i + 1).sum()
    return area

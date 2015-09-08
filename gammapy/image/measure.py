# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Measure image properties.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from gammapy.image.utils import coordinates

__all__ = [
    'BoundingBox',
    'bbox',
    'find_max',
    'lookup',
    'lookup_max',
    'measure_containment_fraction',
    'measure_containment_radius',
    'measure_image_moments',
    'measure_labeled_regions',
    'measure_containment',
    'measure_curve_of_growth'
]


class BoundingBox(object):
    """Rectangular bounding box.

    TODO: Link to bounding box docs.

    Parameters
    ----------
    x_start, y_start : float
        Start pixel coordinates (0-based indexing, inclusive)
    x_stop, y_stop : float
        Stop pixel coordinates (0-based indexing, exclusive)
    """

    def __init__(self, x_start, y_start, x_stop, y_stop):
        self.x_start = x_start
        self.y_start = y_start
        self.x_stop = x_stop
        self.y_stop = y_stop

    @classmethod
    def for_circle(cls, x, y, radius):
        """Compute bounding box for a circle.

        Parameters
        ----------
        x, y : float
            Circle center position
        radius : float
            Circle radius (pix)

        Returns
        -------
        bounding_box : BoundingBox
            Bounding box
        """
        x, y, radius = int(x), int(y), int(radius)
        x_start = x - radius
        y_start = y - radius
        x_stop = x + radius
        y_stop = y + radius
        return cls(x_start, y_start, x_stop, y_stop)

    def __str__(self):
        ss = 'x = ({x_start}, {x_stop}), '.format(**self)
        ss += 'y = ({y_start}, {y_stop}) '.format(**self)
        return ss

    def intersection(self, other):
        """Compute intersection of two bounding boxes.

        Parameters
        ----------
        other : `BoundingBox`
            Other bounding box.

        Returns
        -------
        intersection : `BoundingBox`
            New intersection bounding box

        Examples
        --------
        >>> b1 = BoundingBox(1, 5, 10, 11)
        >>> b2 = BoundingBox(2, 4, 13, 9)
        >>> b1.intersection(b2)
        TODO
        """
        x_start = max(self.x_start, other.x_start)
        x_stop = min(self.x_stop, other.x_stop)
        y_start = max(self.y_start, other.y_start)
        y_stop = min(self.x_stop, other.x_stop)
        return BoundingBox(x_start, y_start, x_stop, y_stop)

    def overlaps(self, other):
        """Does this bounding box overlap the other bounding box?

        Parameters
        ----------
        other : `BoundingBox`
            Other bounding box

        Returns
        -------
        overlaps : bool
            True of there is overlap

        Examples
        --------
        TODO
        """
        return self.intersection(other).is_empty

    def __eq__(self, other):
        return self.slice == other.slice

    @property
    def ftcopy_string(self):
        """Bounding box in ftcopy string format (str)

        The output is given as [x_start:x_stop,y_start:y_stop]

        Examples
        --------
        >>> from gammapy.image import BoundingBox
        >>> bounding_box = BoundingBox(7, 9, 43, 44)
        >>> bounding_box.ftcopy_string
        TODO
        """
        return '[{x_start}:{x_stop},{y_start}:{y_stop}]'.format(**self)

    @property
    def slice(self):
        """Bounding box in slice format (y, x) (tuple)

        Examples
        --------
        >>> from gammapy.image import BoundingBox
        >>> bounding_box = BoundingBox(7, 9, 43, 44)
        >>> bounding_box.slice
        TODO
        """
        return (self.y_slice, self.x_slice)

    @property
    def x_slice(self):
        """Bounding box X component in slice format (slice)"""
        return slice(self.x_start, self.x_stop)

    @property
    def y_slice(self):
        """Bounding box Y component in slice format (slice)"""
        return slice(self.y_start, self.y_stop)

    @property
    def is_empty(self):
        """Check if the bounding box is empty (bool)"""
        x_is_empty = (self.x_start >= self.x_stop)
        y_is_empty = (self.y_start >= self.y_stop)
        return x_is_empty or y_is_empty


def bbox(mask, margin, binsz):
    """Determine the bounding box of a mask.

    TODO: this is an old utility function ... put it into the BoundingBox class.
    """
    from scipy.ndimage.measurements import find_objects
    box = find_objects(mask.astype(int))[0]
    ny, nx = mask.shape
    xmin = max(0, int(box[1].start - margin / binsz)) + 1
    xmax = min(nx - 1, int(box[1].stop + margin / binsz)) + 1
    ymin = max(0, int(box[0].start - margin / binsz)) + 1
    ymax = min(ny - 1, int(box[0].stop + margin / binsz)) + 1
    box_string = '[{xmin}:{xmax},{ymin}:{ymax}]'.format(**locals())
    box = xmin, xmax, ymin, ymax
    return box, box_string


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
    image : `~astropy.io.fits.ImageHDU`
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
    origin = 0  # convention for gammapy
    GLON, GLAT = proj.wcs_pix2world(x, y, origin)
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
    origin = 0  # convention for gammapy
    x, y = wcs.wcs_world2pix(lon, lat, origin)
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
        return _lookup_pix(image.data, x, y)


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
            val[ii] = image.data[mask].max()
        except ValueError:
            pass
    return val


def measure_image_moments(image):
    """Compute 0th, 1st and 2nd moments of an image.

    NaN values are ignored in the computation.

    Parameters
    ----------
    image :`astropy.io.fits.ImageHDU`
        Image to measure on.

    Returns
    -------
    image moments : list
        List of image moments:
        [A, x_cms, y_cms, x_sigma, y_sigma, sqrt(x_sigma * y_sigma)]
    """
    x, y = coordinates(image, lon_sym=True)
    A = image.data[np.isfinite(image.data)].sum()

    # Center of mass
    x_cms = (x * image.data)[np.isfinite(image.data)].sum() / A
    y_cms = (y * image.data)[np.isfinite(image.data)].sum() / A

    # Second moments
    x_var = ((x - x_cms) ** 2 * image.data)[np.isfinite(image.data)].sum() / A
    y_var = ((y - y_cms) ** 2 * image.data)[np.isfinite(image.data)].sum() / A
    x_sigma = np.sqrt(x_var)
    y_sigma = np.sqrt(y_var)

    return A, x_cms, y_cms, x_sigma, y_sigma, np.sqrt(x_sigma * y_sigma)


def measure_containment(image, glon, glat, radius):
    """
    Measure containment in a given circle around the source position.

    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
        Image to measure on.
    glon : float
        Source longitude in degree.
    glat : float
        Source latitude in degree.
    radius : float
        Radius of the region to measure the containment in.
    """
    GLON, GLAT = coordinates(image, lon_sym=True)
    rr = (GLON - glon) ** 2 + (GLAT - glat) ** 2
    return measure_containment_fraction(radius, rr, image.data)


def measure_containment_radius(image, glon, glat, containment_fraction=0.8):
    """Measure containment radius.

    Uses `scipy.optimize.brentq`.

    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
        Image to measure on.
    glon : float
        Source longitude in degree.
    glat : float
        Source latitude in degree.
    containment_fraction : float (default 0.8)
        Containment fraction

    Returns
    -------
    containment_radius : float
        Containment radius (pix)
    """
    from scipy.optimize import brentq

    GLON, GLAT = coordinates(image, lon_sym=True)
    rr = (GLON - glon) ** 2 + (GLAT - glat) ** 2

    # Normalize image
    image.data = image.data / image.data[np.isfinite(image.data)].sum()

    def func(r):
        return measure_containment_fraction(r, rr, image.data) - containment_fraction

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


def measure_curve_of_growth(image, glon, glat, r_max=0.2, delta_r=0.01):
    """
    Measure the curve of growth for a given source position.

    The curve of growth is determined by measuring the flux in a circle around
    the source and radius of this circle is increased

    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
        Image to measure on.
    glon : float
        Source longitude in degree.
    glat : float
        Source latitude in degree.
    r_max : float (default 0.2)
        Maximal radius, up to which the containment is measured in degree.
    delta_r : int (default 10)
        Stepsize for the radius grid in degree.

    Returns
    -------
    radii : array
        Radii where the containment was measured.
    containment : array
        Corresponding contained flux.
    """
    containment = []
    radii = np.arange(0, r_max, delta_r)
    for radius in radii:
        containment.append(measure_containment(image, glon, glat, radius))
    return radii, np.array(containment)


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

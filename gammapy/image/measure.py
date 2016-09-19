# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Measure image properties.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity

__all__ = [
    'measure_containment_fraction',
    'measure_containment_radius',
    'measure_image_moments',
    'measure_labeled_regions',
    'measure_containment',
    'measure_curve_of_growth'
]


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


def _wrapped_coordinates(image):
    coords = image.coordinates()
    return coords.data.lon.wrap_at('180d'), coords.data.lat


def measure_image_moments(image):
    """
    Compute 0th, 1st and 2nd moments of an image.

    NaN values are ignored in the computation.

    Parameters
    ----------
    image : `gammapy.image.SkyImage`
        Image to measure on.

    Returns
    -------
    image moments : list
        List of image moments:
        [A, x_cms, y_cms, x_sigma, y_sigma, sqrt(x_sigma * y_sigma)]
    """
    x, y = _wrapped_coordinates(image)
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


def measure_containment(image, position, radius):
    """
    Measure containment in a given circle around the source position.

    Parameters
    ----------
    image :`gammapy.image.SkyImage`
        Image to measure on.
    position : `~astropy.coordinates.SkyCoord`
        Source position on the sky.
    radius : float
        Radius of the region to measure the containment in.
    """
    separation = image.coordinates().separation(position)
    return measure_containment_fraction(image.data, radius, separation)


def measure_containment_radius(image, position, containment_fraction=0.8):
    """
    Measure containment radius of a source.

    Uses `scipy.optimize.brentq`.

    Parameters
    ----------
    image :`gammapy.image.SkyImage`
        Image to measure on.
    position : `~astropy.coordinates.SkyCoord`
        Source position on the sky.
    containment_fraction : float (default 0.8)
        Containment fraction

    Returns
    -------
    containment_radius :
        Containment radius (pix)
    """
    from scipy.optimize import brentq

    separation = image.coordinates().separation(position)

    # Normalize image
    data = image.data / image.data[np.isfinite(image.data)].sum()

    def func(r):
        return measure_containment_fraction(data, r, separation.value) - containment_fraction

    containment_radius = brentq(func, a=0, b=separation.max().value)
    return Quantity(containment_radius, separation.unit)


def measure_containment_fraction(image, radius, separation):
    """Measure containment fraction.

    Parameters
    ----------
    image :`gammapy.image.SkyImage`
        Image to measure on.
    radius : `~astropy.units.Quantity`
        Containment radius.
    separation : `~astropy.coordinates.Angle`
         Separation from the source position array.

    Returns
    -------
    containment_fraction : float
        Containment fraction
    """
    # Set up indices and containment mask
    containment_mask = separation < radius
    mask = np.logical_and(np.isfinite(image), containment_mask)
    containment_fraction = image[mask].sum()
    return containment_fraction


def measure_curve_of_growth(image, position, radius_max=None, radius_n=10):
    """
    Measure the curve of growth for a given source position.

    The curve of growth is determined by measuring the flux in a circle around
    the source and radius of this circle is increased

    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
        Image to measure on.
    position : `~astropy.coordinates.SkyCoord`
        Source position on the sky.
    radius_max : `~astropy.units.Quantity`
        Maximal radius, up to which the containment is measured (default 0.2 deg).
    radius_n : int
        Number of radius steps.

    Returns
    -------
    radii : `~astropy.units.Quantity`
        Radii where the containment was measured.
    containment : `~astropy.units.Quantity`
        Corresponding contained flux.
    """
    radius_max = radius_max or Quantity(0.2, 'deg')
    containment = []
    radii = Quantity(np.linspace(0, radius_max.value, radius_n), radius_max.unit)
    for radius in radii:
        containment.append(measure_containment(image, position, radius))
    return radii, Quantity(containment)


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

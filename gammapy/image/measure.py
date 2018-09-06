# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity

__all__ = [
    "measure_containment_fraction",
    "measure_containment_radius",
    "measure_image_moments",
    "measure_containment",
    "measure_curve_of_growth",
]


def measure_image_moments(image):
    """
    Compute 0th, 1st and 2nd moments of an image.

    NaN values are ignored in the computation.

    Parameters
    ----------
    image : `gammapy.maps.Map`
        Image to measure on.

    Returns
    -------
    image moments : list
        List of image moments:
        [A, x_cms, y_cms, x_sigma, y_sigma, sqrt(x_sigma * y_sigma)]
    """
    data = image.quantity

    coords = image.geom.get_coord().skycoord
    x, y = coords.data.lon.wrap_at("180d"), coords.data.lat

    A = data[np.isfinite(data)].sum()

    # Center of mass
    x_cms = (x * data)[np.isfinite(data)].sum() / A
    y_cms = (y * data)[np.isfinite(data)].sum() / A

    # Second moments
    x_var = ((x - x_cms) ** 2 * data)[np.isfinite(data)].sum() / A
    y_var = ((y - y_cms) ** 2 * data)[np.isfinite(data)].sum() / A
    x_sigma = np.sqrt(x_var)
    y_sigma = np.sqrt(y_var)

    return A, x_cms, y_cms, x_sigma, y_sigma, np.sqrt(x_sigma * y_sigma)


def measure_containment(image, position, radius):
    """
    Measure containment in a given circle around the source position.

    Parameters
    ----------
    image :`gammapy.maps.Map`
        Image to measure on.
    position : `~astropy.coordinates.SkyCoord`
        Source position on the sky.
    radius : float
        Radius of the region to measure the containment in.
    """
    coords = image.geom.get_coord()
    separation = coords.skycoord.separation(position)
    return measure_containment_fraction(image.quantity, radius, separation)


def measure_containment_radius(image, position, containment_fraction=0.8):
    """
    Measure containment radius of a source.

    Uses `scipy.optimize.brentq`.

    Parameters
    ----------
    image :`gammapy.maps.Map`
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

    data = image.quantity
    coords = image.geom.get_coord()
    separation = coords.skycoord.separation(position)

    # Normalize image
    data = data / data[np.isfinite(data)].sum()

    def func(r):
        return (
            measure_containment_fraction(data, r, separation.value)
            - containment_fraction
        )

    containment_radius = brentq(func, a=0, b=separation.max().value)
    return Quantity(containment_radius, separation.unit)


def measure_containment_fraction(data, radius, separation):
    """Measure containment fraction.

    Parameters
    ----------
    data :`~astropy.unit.Quantity`
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
    mask = np.isfinite(data) & containment_mask
    containment_fraction = data[mask].sum()
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
    radius_max = radius_max if radius_max is not None else Quantity(0.2, "deg")
    containment = []
    radii = Quantity(np.linspace(0, radius_max.value, radius_n), radius_max.unit)
    for radius in radii:
        containment.append(measure_containment(image, position, radius))
    return radii, Quantity(containment)

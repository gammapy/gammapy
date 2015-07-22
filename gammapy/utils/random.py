# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Random sampling for some common distributions"""
from __future__ import print_function, division
import numbers
import numpy as np
from astropy.coordinates import Angle

__all__ = ['check_random_state',
           'sample_sphere',
           'sample_sphere_distance',
           'sample_powerlaw',
           ]

def check_random_state(seed):
    """Turn seed into a `numpy.random.RandomState` instance.

    * If seed is None, return the RandomState singleton used by np.random.
    * If seed is an int, return a new RandomState instance seeded with seed.
    * If seed is already a RandomState instance, return it.
    * Otherwise raise ValueError.

    This function was copied from scikit-learn.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def sample_sphere(size, lon_range=None, lat_range=None):
    """Sample random points on the sphere.

    Reference: http://mathworld.wolfram.com/SpherePointPicking.html

    Parameters
    ----------
    size : int
        Number of samples to generate
    lon_range : `~astropy.coordinates.Angle`, optional
        Longitude range (min, max) in range (0, 360) deg
    lat_range : `~astropy.coordinates.Angle`, optional
        Latitude range (min, max) in range (-90, 90) deg

    Returns
    -------
    lon, lat: `~astropy.units.Angle`
        Longitude and latitude coordinate arrays
    """
    # Convert inputs to internal format (all radians)
    size = int(size)

    if lon_range is not None:
        lon_unit = lon_range.unit
        lon_range.to('radian')
    else:
        lon_unit = 'radian'
        lon_range = Angle([0., 2*np.pi], lon_unit)

    if lat_range is not None:
        lat_unit = lon_range.unit
        lat_range.to('radian')
    else:
        lat_unit = 'radian'
        lat_range = Angle([-np.pi/2., np.pi/2.], lat_unit)

    # Sample random longitude
    u = np.random.random(size)
    lon = lon_range[0] + (lon_range[1] - lon_range[0]) * u

    # Sample random latitude
    v = np.random.random(size)
    #z_range = np.sin(np.array(lat_range))
    z_range = np.sin(lat_range)
    z = z_range[0] + (z_range[1] - z_range[0]) * v
    # This is not the formula given in the reference, but it is equivalent.
    lat = np.arcsin(z)

    # Return result
    return lon.to(lon_unit), lat.to(lat_unit)


def sample_powerlaw(x_min, x_max, gamma, size=None):
    """Sample random values from a power law distribution.

    f(x) = x ** (-gamma) in the range x_min to x_max

    Reference: http://mathworld.wolfram.com/RandomNumber.html

    Parameters
    ----------
    x_min : float
        x range minimum
    x_max : float
        x range maximum
    gamma : float
        Power law index
    size : int
        Number of samples to generate

    Returns
    -------
    x : array
        Array of samples from the distribution
    """
    size = int(size)

    u = np.random.random(size)
    exp = 1. - gamma
    base = x_min ** exp + u * (x_max ** exp - x_min ** exp)
    x = base ** (1 / exp)

    return x


def sample_sphere_distance(distance_min=0, distance_max=1, size=None):
    """Sample random distances if the 3-dim space density is constant.

    This function uses inverse transform sampling
    (`Wikipedia <http://en.wikipedia.org/wiki/Inverse_transform_sampling>`__)
    to generate random distances for an observer located in a 3-dim
    space with constant source density in the range ``(distance_min, distance_max)``.

    Parameters
    ----------
    size : int
        Number of samples
    distance_min, distance_max : float
        Distance range in which to sample

    Returns
    -------
    distance : array
        Array of samples
    """
    # Since the differential distribution is dP / dr ~ r ^ 2,
    # we have a cumulative distribution
    #     P(r) = a * r ^ 3 + b
    # with P(r_min) = 0 and P(r_max) = 1 implying
    #     a = 1 / (r_max ^ 3 - r_min ^ 3)
    #     b = -a * r_min ** 3

    a = 1. / (distance_max ** 3 - distance_min ** 3)
    b = - a * distance_min ** 3

    # Now for inverse transform sampling we need to use the inverse of
    #     u = a * r ^ 3 + b
    # which is
    #     r = [(u - b)/ a] ^ (1 / 3)
    u = np.random.random(size)
    distance = ((u - b) / a) ** (1. / 3)

    return distance

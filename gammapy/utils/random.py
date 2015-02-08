# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Random sampling for some common distributions"""
from __future__ import print_function, division
import numpy as np

__all__ = ['sample_sphere',
           'sample_sphere_distance',
           'sample_powerlaw',
           ]


def sample_sphere(size, lon_range=None, lat_range=None, unit='radians'):
    """Sample random points on the sphere.

    Reference: http://mathworld.wolfram.com/SpherePointPicking.html

    Parameters
    ----------
    size : int
        Number of samples to generate
    lon_range : tuple
        Longitude range (min, max) tuple in range (0, 360)
    lat_range : tuple
        Latitude range (min, max) tuple in range (-180, 180)
    units : {'rad', 'deg'}
        Units of input range and returned angles

    Returns
    -------
    lon, lat: arrays
        Longitude and latitude coordinate arrays
    """
    # Convert inputs to internal format (all radians)
    size = int(size)

    if (lon_range != None) and (unit == 'deg'):
        lon_range = np.radians(lon_range)
    else:
        lon_range = 0, 2 * np.pi

    if (lat_range != None) and (unit == 'deg'):
        lat_range = np.radians(lat_range)
    else:
        lat_range = -np.pi / 2, np.pi / 2

    # Sample random longitude
    u = np.random.random(size)
    lon = lon_range[0] + (lon_range[1] - lon_range[0]) * u

    # Sample random latitude
    v = np.random.random(size)
    z_range = np.sin(np.array(lat_range))
    z = z_range[0] + (z_range[1] - z_range[0]) * v
    # This is not the formula given in the reference, but it is equivalent.
    lat = np.arcsin(z)

    # Return result
    if unit in ['rad', 'radian', 'radians']:
        return lon, lat
    elif unit in ['deg', 'degree', 'degrees']:
        return np.degrees(lon), np.degrees(lat)
    else:
        raise ValueError('Invalid unit: {0}'.format(unit))


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


def sample_sphere_distance(distance_min, distance_max, size=None):
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
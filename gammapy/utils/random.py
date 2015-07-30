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


def sample_sphere(size, lon_range=None, lat_range=None, random_state=None):
    """Sample random points on the sphere.

    Reference: http://mathworld.wolfram.com/SpherePointPicking.html

    Parameters
    ----------
    size : int
        Number of samples to generate
    lon_range : `~astropy.coordinates.Angle`, optional
        Longitude range (min, max)
    lat_range : `~astropy.coordinates.Angle`, optional
        Latitude range (min, max)
    random_state : int or `~numpy.random.RandomState`, optional
        Pseudo-random number generator state used for random
        sampling. Separate function calls with the same parameters
        and ``random_state`` will generate identical results.

    Returns
    -------
    lon, lat: `~astropy.units.Angle`
        Longitude and latitude coordinate arrays
    """
    # initialise random number generator
    rng = check_random_state(random_state)

    #Check input parameters
    if lon_range is None:
        lon_range = Angle([0., 360.], 'degree')

    if lat_range is None:
        lat_range = Angle([-90., 90.], 'degree')

    # Sample random longitude
    u = rng.uniform(size=size)
    lon = lon_range[0] + (lon_range[1] - lon_range[0]) * u

    # Sample random latitude
    v = rng.uniform(size=size)
    z_range = np.sin(lat_range)
    z = z_range[0] + (z_range[1] - z_range[0]) * v
    lat = np.arcsin(z)

    # Return result
    return lon, lat


def sample_powerlaw(x_min, x_max, gamma, size=None, random_state=None):
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
    size : int, optional
        Number of samples to generate
    random_state : int or `~numpy.random.RandomState`, optional
        Pseudo-random number generator state used for random
        sampling. Separate function calls with the same parameters
        and ``random_state`` will generate identical results.

    Returns
    -------
    x : array
        Array of samples from the distribution
    """
    # initialise random number generator
    rng = check_random_state(random_state)

    size = int(size)

    exp = 1. - gamma
    base = rng.uniform(x_min ** exp, x_max ** exp, size)
    x = base ** (1 / exp)

    return x


def sample_sphere_distance(distance_min=0, distance_max=1, size=None, random_state=None):
    """Sample random distances if the 3-dim space density is constant.

    This function uses inverse transform sampling
    (`Wikipedia <http://en.wikipedia.org/wiki/Inverse_transform_sampling>`__)
    to generate random distances for an observer located in a 3-dim
    space with constant source density in the range ``(distance_min, distance_max)``.

    Parameters
    ----------
    distance_min, distance_max : float, optional
        Distance range in which to sample
    size : int, optional
        Number of samples
    random_state : int or `~numpy.random.RandomState`, optional
        Pseudo-random number generator state used for random
        sampling. Separate function calls with the same parameters
        and ``random_state`` will generate identical results.

    Returns
    -------
    distance : array
        Array of samples
    """
    # initialise random number generator
    rng = check_random_state(random_state)

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
    u = rng.uniform(size=size)
    distance = ((u - b) / a) ** (1. / 3)

    return distance

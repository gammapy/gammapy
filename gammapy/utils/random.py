# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Random sampling for some common distributions"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numbers
import numpy as np
from astropy.coordinates import Angle

__all__ = [
    "get_random_state",
    "sample_sphere",
    "sample_sphere_distance",
    "sample_powerlaw",
]


def get_random_state(init):
    """Get a `numpy.random.RandomState` instance.

    The purpose of this utility function is to have a flexible way
    to initialise a `~numpy.random.RandomState` instance,
    a.k.a. a random number generator (``rng``).

    See :ref:`dev_random` for usage examples and further information.

    Parameters
    ----------
    init : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Available options to initialise the RandomState object:

        * ``int`` -- new RandomState instance seeded with this integer
          (calls `~numpy.random.RandomState` with ``seed=init``)
        * ``'random-seed'`` -- new RandomState instance seeded in a random way
          (calls `~numpy.random.RandomState` with ``seed=None``)
        * ``'global-rng'``, return the RandomState singleton used by ``numpy.random``.
        * `~numpy.random.RandomState` -- do nothing, return the input.

    Returns
    -------
    random_state : `~numpy.random.RandomState`
        RandomState instance.
    """
    if isinstance(init, (numbers.Integral, np.integer)):
        return np.random.RandomState(init)
    elif init == "random-seed":
        return np.random.RandomState(None)
    elif init == "global-rng":
        return np.random.mtrand._rand
    elif isinstance(init, np.random.RandomState):
        return init
    else:
        raise ValueError(
            "{} cannot be used to seed a numpy.random.RandomState"
            " instance".format(init)
        )


def sample_sphere(size, lon_range=None, lat_range=None, random_state="random-seed"):
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
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    lon, lat: `~astropy.coordinates.Angle`
        Longitude and latitude coordinate arrays
    """
    random_state = get_random_state(random_state)

    if lon_range is None:
        lon_range = Angle([0., 360.], "deg")

    if lat_range is None:
        lat_range = Angle([-90., 90.], "deg")

    # Sample random longitude
    u = random_state.uniform(size=size)
    lon = lon_range[0] + (lon_range[1] - lon_range[0]) * u

    # Sample random latitude
    v = random_state.uniform(size=size)
    z_range = np.sin(lat_range)
    z = z_range[0] + (z_range[1] - z_range[0]) * v
    lat = np.arcsin(z)

    return lon, lat


def sample_powerlaw(x_min, x_max, gamma, size=None, random_state="random-seed"):
    """Sample random values from a power law distribution.

    f(x) = x ** (-gamma) in the range x_min to x_max

    It is assumed that *gamma* is the **differential** spectral index.

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
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    x : array
        Array of samples from the distribution
    """
    random_state = get_random_state(random_state)

    size = int(size)

    exp = -gamma
    base = random_state.uniform(x_min ** exp, x_max ** exp, size)
    x = base ** (1 / exp)

    return x


def sample_sphere_distance(
    distance_min=0, distance_max=1, size=None, random_state="random-seed"
):
    """Sample random distances if the 3-dim space density is constant.

    This function uses inverse transform sampling
    (`Wikipedia <http://en.wikipedia.org/wiki/Inverse_transform_sampling>`__)
    to generate random distances for an observer located in a 3-dim
    space with constant source density in the range ``(distance_min, distance_max)``.

    Parameters
    ----------
    distance_min, distance_max : float, optional
        Distance range in which to sample
    size : int or tuple of ints, optional
        Output shape. Default: one sample. Passed to `numpy.random.uniform`.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    distance : array
        Array of samples
    """
    random_state = get_random_state(random_state)

    # Since the differential distribution is dP / dr ~ r ^ 2,
    # we have a cumulative distribution
    #     P(r) = a * r ^ 3 + b
    # with P(r_min) = 0 and P(r_max) = 1 implying
    #     a = 1 / (r_max ^ 3 - r_min ^ 3)
    #     b = -a * r_min ** 3

    a = 1. / (distance_max ** 3 - distance_min ** 3)
    b = -a * distance_min ** 3

    # Now for inverse transform sampling we need to use the inverse of
    #     u = a * r ^ 3 + b
    # which is
    #     r = [(u - b)/ a] ^ (1 / 3)
    u = random_state.uniform(size=size)
    distance = ((u - b) / a) ** (1. / 3)

    return distance

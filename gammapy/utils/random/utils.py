# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions to work with distributions."""
import numbers
import numpy as np
import scipy.integrate
from astropy.coordinates import Angle
from astropy.time import TimeDelta

__all__ = [
    "draw",
    "get_random_state",
    "normalize",
    "pdf",
    "sample_powerlaw",
    "sample_sphere",
    "sample_sphere_distance",
    "sample_times",
]


def normalize(func, x_min, x_max):
    """Normalize a 1D function over a given range."""

    def f(x):
        return func(x) / scipy.integrate.quad(func, x_min, x_max)[0]

    return f


def pdf(func):
    """One-dimensional PDF of a given radial surface density."""

    def f(x):
        return x * func(x)

    return f


def draw(low, high, size, dist, random_state="random-seed", *args, **kwargs):
    """Allows drawing of random numbers from any distribution."""
    from .inverse_cdf import InverseCDFSampler

    n = 1000
    x = np.linspace(low, high, n)

    pdf = dist(x)
    sampler = InverseCDFSampler(pdf=pdf, random_state=random_state)

    idx = sampler.sample(size)
    x_sampled = np.interp(idx, np.arange(n), x)
    return np.squeeze(x_sampled)


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
        lon_range = Angle([0.0, 360.0], "deg")

    if lat_range is None:
        lat_range = Angle([-90.0, 90.0], "deg")

    # Sample random longitude
    u = random_state.uniform(size=size)
    lon = lon_range[0] + (lon_range[1] - lon_range[0]) * u

    # Sample random latitude
    v = random_state.uniform(size=size)
    z_range = np.sin(lat_range)
    z = z_range[0] + (z_range[1] - z_range[0]) * v
    lat = np.arcsin(z)

    return lon, lat


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

    a = 1.0 / (distance_max**3 - distance_min**3)
    b = -a * distance_min**3

    # Now for inverse transform sampling we need to use the inverse of
    #     u = a * r ^ 3 + b
    # which is
    #     r = [(u - b)/ a] ^ (1 / 3)
    u = random_state.uniform(size=size)
    distance = ((u - b) / a) ** (1.0 / 3)

    return distance


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
    base = random_state.uniform(x_min**exp, x_max**exp, size)
    x = base ** (1 / exp)

    return x


def sample_times(
    size,
    rate,
    dead_time=TimeDelta(0, format="sec"),
    return_diff=False,
    random_state="random-seed",
):
    """Make random times assuming a Poisson process.

    This function can be used to test event time series,
    to have a comparison what completely random data looks like.

    Can be used in two ways (in either case the return type is `~astropy.time.TimeDelta`):

    * ``return_delta=False`` - Return absolute times, relative to zero (default)
    * ``return_delta=True`` - Return time differences between consecutive events.

    Parameters
    ----------
    size : int
        Number of samples
    rate : `~astropy.units.Quantity`
        Event rate (dimension: 1 / TIME)
    dead_time : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`, optional
        Dead time after event (dimension: TIME)
    return_diff : bool
        Return time difference between events? (default: no, return absolute times)
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    time : `~astropy.time.TimeDelta`
        Time differences (second) after time zero.

    Examples
    --------
    Example how to simulate 100 events at a rate of 10 Hz.
    As expected the last event occurs after about 10 seconds.

    >>> from astropy.units import Quantity
    >>> from gammapy.utils.random import sample_times
    >>> rate = Quantity(10, 'Hz')
    >>> times = sample_times(size=100, rate=rate, random_state=0)
    >>> times[-1]
    <TimeDelta object: scale='None' format='sec' value=9.186484131475074>
    """
    random_state = get_random_state(random_state)

    dead_time = TimeDelta(dead_time)
    scale = (1 / rate).to("s").value
    time_delta = random_state.exponential(scale=scale, size=size)
    time_delta += dead_time.to("s").value

    if return_diff:
        return TimeDelta(time_delta, format="sec")
    else:
        time = time_delta.cumsum()
        return TimeDelta(time, format="sec")

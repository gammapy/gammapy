"""Helper functions to work with distributions."""
from __future__ import print_function, division
from ...utils.distributions import GeneralRandom

__all__ = ['normalize', 'density', 'draw', 'pdf']


def normalize(func, x_min, x_max):
    """Normalize a 1D function over a given range.
    """
    from scipy.integrate import quad

    def f(x):
        return func(x) / quad(func, x_min, x_max)[0]

    return f


def pdf(func):
    """Returns the one dimensional PDF of a given radial surface density.
    """
    def f(x):
        return x * func(x)

    return f


def density(func):
    """Returns the radial surface density of a given one dimensional PDF.
    """
    def f(x):
        return func(x) / x

    return f


def draw(low, high, size, dist, *args, **kwargs):
    """Allows drawing of random numbers from any distribution."""
    def f(x):
        return dist(x, *args, **kwargs)

    d = GeneralRandom(f, low, high)
    array = d.draw(size)
    return array

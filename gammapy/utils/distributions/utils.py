# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions to work with distributions."""
from __future__ import print_function, division

__all__ = ['normalize', 'density', 'draw', 'pdf']


def normalize(func, x_min, x_max):
    """Normalize a 1D function over a given range.
    """
    from scipy.integrate import quad
    normalized_func = lambda x: func(x) / quad(func, x_min, x_max)[0]
    return normalized_func


def pdf(func):
    """Returns the one dimensional PDF of a given radial surface density.
    """
    return lambda x: x * func(x)


def density(func):
    """Returns the radial surface density of a given one dimensional PDF.
    """
    return lambda x: func(x) / x


def draw(low, high, size, dist, *args, **kwargs):
    """Allows drawing of random numbers from any distribution."""
    from .general_random import GeneralRandom
    f = lambda x: dist(x, *args, **kwargs)
    d = GeneralRandom(f, low, high)
    return d.draw(size)

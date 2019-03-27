# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions to work with distributions."""
from scipy.integrate import quad
from ...utils.random import get_random_state
from ...utils.distributions import GeneralRandom

__all__ = ["normalize", "density", "draw", "pdf"]


def normalize(func, x_min, x_max):
    """Normalize a 1D function over a given range."""

    def f(x):
        return func(x) / quad(func, x_min, x_max)[0]

    return f


def pdf(func):
    """One-dimensional PDF of a given radial surface density."""

    def f(x):
        return x * func(x)

    return f


def density(func):
    """Radial surface density of a given one dimensional PDF."""

    def f(x):
        return func(x) / x

    return f


def draw(low, high, size, dist, random_state="random-seed", *args, **kwargs):
    """Allows drawing of random numbers from any distribution."""
    random_state = get_random_state(random_state)

    def f(x):
        return dist(x, *args, **kwargs)

    d = GeneralRandom(f, low, high)
    return d.draw(size, random_state=random_state)

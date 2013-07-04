# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""On-off bin stats computations"""
import numpy as np

__all__ = ['Stats', 'SpectrumStats', 'make_stats', 'combine_stats']


class Stats(object):
    """Contains count info"""

    def __init__(self, n_on, n_off, e_on, e_off):
        self.n_on = float(n_on)
        self.n_off = float(n_off)
        self.e_on = float(e_on)
        self.e_off = float(e_off)

    def alpha(self):
        return self.e_on / self.e_off

    def background(self):
        return self.alpha() * self.n_off

    def excess(self):
        return self.n_on - self.background()

    def __str__(self):
        keys = ['n_on', 'n_off', 'e_on', 'e_off',
                'alpha', 'background', 'excess']
        values = [self.n_on, self.n_off, self.e_on, self.e_off,
                  self.alpha(), self.background(), self.excess()]
        return '\n'.join(['%s = %s' % (k, v)
                          for (k, v) in zip(keys, values)])


class SpectrumStats(Stats):

    """Contains count and exposure info"""
    # TODO: implement me!
    pass


def make_stats(signal, background, area_factor, weight_method="background",
               poisson_fluctuate=False):
    """Fill using some weight method for the exposure"""
    # Compute counts
    n_on = signal + background
    n_off = area_factor * background
    if poisson_fluctuate:
        n_on = np.random.poisson(n_on)
        n_off = np.random.poisson(n_off)

    # Compute weight
    if weight_method == "none":
        weight = 1
    elif weight_method == "background":
        weight = background
    elif weight_method == "n_off":
        weight = n_off
    else:
        raise ValueError("Invalid weight_method: {0}".format(weight_method))

    # Compute exposure
    e_on = weight
    e_off = weight * area_factor
    return Stats(n_on, n_off, e_on, e_off)


def combine_stats(stats_1, stats_2, weight_method="none"):
    """Combine using some weight method for the exposure."""
    # Compute counts
    n_on = stats_1.n_on + stats_2.n_on
    n_off = stats_1.n_off + stats_2.n_off

    # Compute weights
    if weight_method == "none":
        weight_1 = 1
        weight_2 = 1
    elif weight_method == "background":
        weight_1 = stats_1.background()
        weight_2 = stats_2.background()
    elif weight_method == "n_off":
        weight_1 = stats_1.n_off
        weight_2 = stats_2.n_off
    else:
        raise ValueError("Invalid weight_method: {0}".format(weight_method))

    # Compute exposure
    e_on = weight_1 * stats_1.e_on + weight_2 * stats_2.e_on
    e_off = weight_1 * stats_1.e_off + weight_2 * stats_2.e_off
    return Stats(n_on, n_off, e_on, e_off)

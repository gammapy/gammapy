# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""On-off bin stats computations."""
from __future__ import print_function, division
import numpy as np

__all__ = ['Stats', 'make_stats', 'combine_stats']


class Stats(object):
    """Container for an on-off observation.
    
    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    a_on : array_like
        Relative background exposure of the on region
    a_off : array_like
        Relative background exposure in the off region
    """
    # TODO: use numpy arrays and properties
    # TODO: add gamma exposure

    def __init__(self, n_on, n_off, a_on, a_off):
        self.n_on = float(n_on)
        self.n_off = float(n_off)
        self.a_on = float(a_on)
        self.a_off = float(a_off)

    @property
    def alpha(self):
        r"""Background exposure ratio.
        
        .. math:: \alpha = a_{on} / a_{off}
        """
        return self.a_on / self.a_off

    @property
    def background(self):
        r"""Background estimate.
        
        .. math:: \mu_{background} = \alpha\ n_{off}
        """
        return self.alpha * self.n_off

    @property
    def excess(self):
        r"""Excess.
        
        .. math:: n_{excess} = n_{on} - \mu_{background}
        """
        return self.n_on - self.background

    def __str__(self):
        keys = ['n_on', 'n_off', 'a_on', 'a_off',
                'alpha', 'background', 'excess']
        values = [self.n_on, self.n_off, self.a_on, self.a_off,
                  self.alpha, self.background(), self.excess()]
        return '\n'.join(['%s = %s' % (k, v)
                          for (k, v) in zip(keys, values)])


def make_stats(signal, background, area_factor, weight_method="background",
               poisson_fluctuate=False):
    """Fill using some weight method for the exposure.
    """
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
    a_on = weight
    a_off = weight * area_factor
    return Stats(n_on, n_off, a_on, a_off)


def combine_stats(stats_1, stats_2, weight_method="none"):
    """Combine using some weight method for the exposure.
    
    Parameters
    ----------
    stats_1 : `Stats`
        Observation 1
    stats_2 : `Stats`
        Observation 2
    weight_method : {'none', 'background', 'n_off'}
        Observation weighting method.

    Returns
    -------
    stats : `Stats`
        Combined Observation 1 and 2
    """
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
    a_on = weight_1 * stats_1.a_on + weight_2 * stats_2.a_on
    a_off = weight_1 * stats_1.a_off + weight_2 * stats_2.a_off

    return Stats(n_on, n_off, a_on, a_off)

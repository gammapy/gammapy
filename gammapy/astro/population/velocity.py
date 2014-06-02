# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pulsar velocity distribution models"""
from __future__ import print_function, division
from numpy import exp, sqrt, pi
from astropy.utils.compat.odict import OrderedDict

__all__ = ['H05', 'F06B', 'F06P',
           'v_range',
           'velocity_distributions',
           ]

# Simulation range used for random number drawing
v_range = 4000  # km/s

def H05(v, sigma=265):
    r"""Maxwellian pulsar velocity distribution.

    .. math ::
        f(v) = \sqrt(2 / \pi) \frac{v ^ 2}{\sigma ^ 3}
               \exp(\frac{-v ^ 2}{2 \sigma ^ 2})

    Reference: TODO: who proposed it?

    Parameters
    ----------
    v : array_like
        Velocity (km s^-1)
    sigma : array_like
        Velocity parameter (km s^-1)

    Returns
    -------
    density : array_like
        Density in velocity ``v``
    """
    term1 = sqrt(2 / pi) * v ** 2 / sigma ** 3
    term2 = exp(-v ** 2 / (2 * sigma ** 2))
    return term1 * term2


def F06B(v, sigma1=160, sigma2=780, w=0.9):
    """Bimodal pulsar velocity distribution - Faucher & Kaspi (2006).

    .. math ::
        f(v) = TODO

    Reference: http://adsabs.harvard.edu/abs/2006ApJ...643..332F

    Parameters
    ----------
    v : array_like
        Velocity (km s^-1)
    sigma1, sigma2 : array_like
        Velocity parameter (km s^-1)
    w : array_like
        See formula

    Returns
    -------
    density : array_like
        Density in velocity ``v``
    """
    term1 = sqrt(2 / pi) * v ** 2 * (w / sigma1 ** 3)
    term2 = exp(-v ** 2 / (2 * sigma1 ** 2))
    term3 = (1 - w) / sigma2 ** 3 * exp(-v ** 2 / (2 * sigma2 ** 2))
    return term1 * term2 + term3


def F06P(v, v0=560):
    """Distribution by Lyne 1982 and adopted by Paczynski and Faucher.
    
    .. math ::
        f(v) = 4 / (\pi v_{0} (1 + (v / v_{0}) ^ 2) ^ 2)
    
    Parameters
    ----------
    v : array_like
        Velocity (km s^-1)
    v0 : array_like
        Velocity parameter (km s^-1)
    
    Returns
    -------
    density : array_like
        Density in velocity ``v``
    """
    return 4. / (pi * v0 * (1 + (v / v0) ** 2) ** 2)

velocity_distributions = OrderedDict()
"""Dictionary of available distributions.

Useful for automatic processing.
"""
velocity_distributions['H05'] = H05
velocity_distributions['F06B'] = F06B
velocity_distributions['F06P'] = F06P


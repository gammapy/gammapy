# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pulsar velocity distribution models"""
from __future__ import print_function, division
from numpy import exp, sqrt, pi

__all__ = ['H05', 'F06B', 'F06P',
           'distributions', 'v_range']

# Simulation range used for random number drawing
v_range = 4000  # km/s

def H05(v, sigma=265):
    """Maxwellian Velocity Distribution.
    @param v: velocity (km s^-1)
    @param sigma : velocity parameter (km s^-1)"""
    return (sqrt(2 / pi) * v ** 2 / sigma ** 3 * 
            exp(-v ** 2 / (2 * sigma ** 2)))


def F06B(v, sigma1=160, sigma2=780, w=0.9):
    """Faucher Bimodal Velocity Distribution.
    Parameters: sigma1=160km/s, sigma2=780km/s, w=0.9"""
    return (sqrt(2 / pi) * v ** 2 * (w / sigma1 ** 3 * 
            exp(-v ** 2 / (2 * sigma1 ** 2)) + 
            (1 - w) / sigma2 ** 3 * exp(-v ** 2 / (2 * sigma2 ** 2))))


def F06P(v, v0=560):
    """Distribution by Lyne 1982 and adopted by Paczynski and Faucher.
    Parameter: v0=560km/s"""
    return 4. / (pi * v0 * (1 + (v / v0) ** 2) ** 2)

# Dictionary of available distributions.
# Useful for automatic processing.
distributions = {'H05': H05, 'F06B': F06B, 'F06P': F06P}

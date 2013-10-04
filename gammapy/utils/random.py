# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Random sampling for some common distributions"""
from __future__ import print_function, division
import numpy as np

__all__ = ['sample_sphere', 'sample_powerlaw']


def sample_sphere(size, unit='radians'):
    """Sample random points on the sphere.
    
    Reference: http://mathworld.wolfram.com/SpherePointPicking.html

    Parameters
    ----------
    size : int
        Number of samples to generate
    units : {'rad', 'deg'}
        Units of returned angles

    Returns
    -------
    lon, lat: arrays
        Longitude and latitude coordinate arrays
    """
    size = int(size)

    u = np.random.random(size)
    lon = 2 * np.pi * u

    v = np.random.random(size)
    lat = np.arccos(2 * v - 1)

    if unit in ['rad', 'radian', 'radians']:
        return lon, lat
    elif unit in ['deg', 'degree', 'degrees']:
        return np.degrees(lon), np.degrees(lat)
    else:
        raise ValueError('Invalid unit: {0}'.format(unit))


def sample_powerlaw(x_min, x_max, gamma, size):
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
    
    size : int
        Number of samples to generate
    
    Returns
    -------
    x : array
        Array of samples from the distribution
    """
    size = int(size)
    
    u = np.random.random(size)
    exp = 1. - gamma
    base = x_min ** exp + u * (x_max ** exp - x_min ** exp)
    x = base ** (1 / exp)
        
    return x

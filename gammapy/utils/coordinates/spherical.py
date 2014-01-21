# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spherical coordinate utility functions.
"""
from __future__ import print_function, division
import numpy as np
from numpy import (cos, sin, arccos, radians, degrees)

__all__ = ['separation', 'minimum_separation', 'pair_correlation']


def separation(lon1, lat1, lon2, lat2, unit='deg'):
    """Angular separation between points on the sphere.
    
    Parameters
    ----------
    lon1, lat1, lon2, lat2 : array_like
        Coordinates of the two points
    unit : {'deg', 'rad'}
        Units of input and output coordinates

    Returns
    -------
    separation : array_like
        Angular separation
    """
    if unit == 'deg':
        lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))

    term1 = cos(lat1) * cos(lon1) * cos(lat2) * cos(lon2)
    term2 = cos(lat1) * sin(lon1) * cos(lat2) * sin(lon2)
    term3 = sin(lat1) * sin(lat2)
    mu = term1 + term2 + term3
    separation = arccos(mu)
    
    if unit == 'deg':
        separation = degrees(separation)

    return separation


def minimum_separation(lon1, lat1, lon2, lat2, unit='deg'):
    """Compute minimum distance of each (lon1, lat1) to any (lon2, lat2).

    Parameters
    ----------
    lon1, lat1 : array_like
        Primary coordinates of interest
    lon2, lat2 : array_like
        Counterpart coordinate array
    unit : {'deg', 'rad'}
        Units of input and output coordinates

    Returns
    -------
    theta_min : array
        Minimum distance
    """
    lon1 = np.asanyarray(lon1)
    lat1 = np.asanyarray(lat1)
    
    theta_min = np.empty_like(lon1, dtype=np.float64)

    for i1 in range(lon1.size):
        thetas = separation(lon1[i1], lat1[i1],
                            lon2, lat2, unit=unit)
        theta_min[i1] = thetas.min()

    return theta_min


def pair_correlation(lon, lat, theta_bins, unit='deg'):
    """Compute pair correlation function for points on the sphere.
    
    Parameters
    ----------
    lon, lat : array_like
        Coordinate arrays
    theta_bins : array_like
        Array defining the `theta` binning.
        `theta` is the angular offset between positions.
    unit : {'deg', 'rad'}
        Units of input and output coordinates

    Returns
    -------
    counts : array
        Array of point separations per `theta` bin.
    """
    # TODO: Implement speedups:
    # - use radians
    # - avoid processing each pair twice (distance a to b and b to a)
    counts = np.zeros(shape=len(theta_bins) - 1, dtype=int)
    # If there are many points this should have acceptable performance
    # because the inner loop is in np.histogram, not in Python
    for ii in range(len(lon)):
        theta = separation(lon[ii], lat[ii], lon, lat, unit=unit)
        hist = np.histogram(theta, theta_bins)[0]
        counts += hist

    return counts

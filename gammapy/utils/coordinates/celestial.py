# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Celestial coordinate utility functions."""
import numpy as np
from astropy.coordinates.angle_utilities import angular_separation

__all__ = ["minimum_separation", "pair_correlation"]


def minimum_separation(lon1, lat1, lon2, lat2):
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
    separation = angular_separation(
        lon1[:, np.newaxis], lat1[:, np.newaxis], lon2, lat2
    )
    return separation.min(axis=1)


def pair_correlation(lon, lat, theta_bins):
    """Compute pair correlation function for points on the sphere.

    Parameters
    ----------
    lon, lat : array_like
        Coordinate arrays
    theta_bins : array_like
        Array defining the ``theta`` binning.
        ``theta`` is the angular offset between positions.
    unit : {'deg', 'rad'}
        Units of input and output coordinates

    Returns
    -------
    counts : array
        Array of point separations per ``theta`` bin.
    """
    separation = angular_separation(lon[:, np.newaxis], lat[:, np.newaxis], lon, lat)
    pair_correlation, _ = np.histogram(separation.ravel(), theta_bins)
    return pair_correlation

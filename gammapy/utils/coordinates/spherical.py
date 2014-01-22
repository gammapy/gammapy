# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spherical coordinate utility functions.
"""
from __future__ import print_function, division
import numpy as np
from numpy import (cos, sin, arccos, radians, degrees)

__all__ = ['separation', 'minimum_separation', 'pair_correlation',
           'pixel_solid_angle']


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
    # - avoid processing each pair twice (distance a to b and b to a)
    counts = np.zeros(shape=len(theta_bins) - 1, dtype=int)
    # If there are many points this should have acceptable performance
    # because the inner loop is in np.histogram, not in Python
    for ii in range(len(lon)):
        theta = separation(lon[ii], lat[ii], lon, lat, unit=unit)
        hist = np.histogram(theta, theta_bins)[0]
        counts += hist

    return counts


def pixel_solid_angle(corners, method='1'):
    """Pixel solid angle on the sphere.

    Parameters
    ----------
    corners : list
            List of dict with `lon` and `lat` keys and array-like values.
    method : {'1', '2'}
        Method to compute the solid angle

    Returns
    -------
    solid_angle : `numpy.array`
        Per-pixel solid angle image in steradians
        
    See also
    --------
    image.utils.solid_angle
    """
    if method == '1':
        return _pixel_solid_angle_1(corners)
    elif method == '2':
        return _pixel_solid_angle_2(corners)
    else:
        raise ValueError('Unknown method: {0}'.format(method))


def _pixel_solid_angle_1(corners):
    """Compute pixel solid angle using 3D cartesian vectors.
    
    And Girard's theorem:
    http://mathworld.wolfram.com/GirardsSphericalExcessFormula.html
    
    Reference: http://mail.scipy.org/pipermail/astropy/2013-December/002940.html
    """
    from astropy.coordinates import spherical_to_cartesian
    
    vec = []
    for corner in corners:
        x, y, z = spherical_to_cartesian(1, corner['lon'], corner['lat'])
        vec.append(dict(x=x, y=y, z=z))

    angles = []
    N = 4
    for i in range(N):
        A = vec[(i + 1) % N]
        B = vec[(i + 2) % N]
        C = vec[(i + 3) % N]
        vec_a = np.cross(A, (np.cross(A, B)))
        vec_b = np.cross(B, (np.cross(C, B)))
        angle = np.arccos(np.dot(vec_a, vec_b))
        angles.append(angle)

    # Use Girard equation for excess area to determine solid angle
    solid_angle = np.sum(angles) - 2 * np.pi

    return solid_angle    


def _pixel_solid_angle_2(corners):
    """Compute pixel solid angle using spherical polygon area formulas.
    
    And Girard's theorem:
    http://mathworld.wolfram.com/GirardsSphericalExcessFormula.html
    
    TODO: possible to replace duplicated code with array expression or loop?
    """
    from astropy.coordinates.angle_utilities import angular_separation

    # Compute angular distances between corners
    def dist(i, j):
        return angular_separation(corners[i-1]['lon'], corners[i-1]['lat'],
                                  corners[j-1]['lon'], corners[j-1]['lat'])
    a12 = dist(1, 2)
    a14 = dist(1, 4)
    a23 = dist(2, 3)
    a34 = dist(3, 4)
    a13 = dist(1, 3)
    a24 = dist(2, 4)
    #for i in range(4):
    #    print(corners[i]['lon'][0, 0], corners[i]['lat'][0, 0])
    #print(a12.shape, a12[0, 0])
    
    # Compute sines and cosines of the corner distance angles
    sin_a12 = np.sin(a12)
    sin_a14 = np.sin(a14)
    sin_a23 = np.sin(a23)
    sin_a34 = np.sin(a34)
    cos_a12 = np.cos(a12)
    cos_a13 = np.cos(a13)
    cos_a14 = np.cos(a14)
    cos_a23 = np.cos(a23)
    cos_a24 = np.cos(a24)
    cos_a34 = np.cos(a34)

    # Compute inner angles
    angle1 = np.arccos((cos_a13 - cos_a34 * cos_a14) / (sin_a34 * sin_a14))
    angle2 = np.arccos((cos_a24 - cos_a23 * cos_a34) / (sin_a23 * sin_a34))
    angle3 = np.arccos((cos_a13 - cos_a12 * cos_a23) / (sin_a12 * sin_a23))
    angle4 = np.arccos((cos_a24 - cos_a14 * cos_a12) / (sin_a14 * sin_a12))

    # Use Girard equation for excess area to determine solid angle
    solid_angle = (angle1 + angle2 + angle3 + angle4) - 2 * np.pi
    
    return solid_angle

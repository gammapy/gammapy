# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Equatorial to / from Galactic coordinate conversion and sky separation.

This is a temporal solution until astropy supports array coordinate computations.

You could also use Kapteyn instead:

* from kapteyn import wcs, maputils
* wcs.Transformation(wcs.equatorial, wcs.galactic)
* wcs.Transformation(wcs.galactic, wcs.equatorial)
* maputils.dist_on_sphere
"""
from __future__ import print_function, division
import numpy as np
from numpy import (cos, sin, arccos, arcsin,
                   arctan2, radians, degrees, pi)

__all__ = ['gal2equ', 'equ2gal', 'separation', 'minimum_separation', 'pair_correlation']


def gal2equ(ll, bb):
    """Converts Galactic to Equatorial J2000 coordinates (deg).

    RA output is in range 0 to 360 deg.
    Only accurate to ~ 3 digits.
    """
    ll, bb = map(radians, (ll, bb))
    ra_gp = radians(192.85948)
    de_gp = radians(27.12825)
    lcp = radians(122.932)
    sin_d = sin(de_gp) * sin(bb) + cos(de_gp) * cos(bb) * cos(lcp - ll)
    ramragp = (arctan2(cos(bb) * sin(lcp - ll),
                       cos(de_gp) * sin(bb) - sin(de_gp) *
                       cos(bb) * cos(lcp - ll)))
    dec = arcsin(sin_d)
    ra = (ramragp + ra_gp + 2 * pi) % (2 * pi)
    ra = ra % 360
    return degrees(ra), degrees(dec)


def equ2gal(ra, dec):
    """Converts Equatorial J2000 to Galactic coordinates (deg).

    RA output is in range 0 to 360 deg.
    Only accurate to ~ 3 digits.
    """
    ra, dec = map(radians, (ra, dec))
    ra_gp = radians(192.85948)
    de_gp = radians(27.12825)
    lcp = radians(122.932)
    sin_b = (sin(de_gp) * sin(dec) + cos(de_gp) *
            cos(dec) * cos(ra - ra_gp))
    lcpml = arctan2(cos(dec) * sin(ra - ra_gp),
                 cos(de_gp) * sin(dec) - sin(de_gp) *
                 cos(dec) * cos(ra - ra_gp))
    bb = arcsin(sin_b)
    ll = (lcp - lcpml + 2 * pi) % (2 * pi)
    ll = ll % 360
    return degrees(ll), degrees(bb)


def separation(lon1, lat1, lon2, lat2):
    """Angular separation in degrees between two sky coordinates
    
    Input and output in degrees.
    """
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
    mu = (cos(lat1) * cos(lon1) * cos(lat2) * cos(lon2)
          + cos(lat1) * sin(lon1) * cos(lat2) * sin(lon2) +
          sin(lat1) * sin(lat2))
    return degrees(arccos(mu))

def minimum_separation(lon1, lat1, lon2, lat2):
    """Compute minimum distance of each (lon1, lat1) to any (lon2, lat2).

    Parameters
    ----------
    lon1, lat1 : array-like
        Primary coordinates of interest in deg
    lon2, lat2 : array-like
        Counterpart coordinate array in deg

    Returns
    -------
    theta_min : array
        Minimum distance in deg
    """
    theta_min = np.empty_like(lon1)

    for i1 in range(lon1.size):
        thetas = separation(lon1[i1], lat1[i1], lon2, lat2)
        theta_min[i1] = thetas.min()

    return theta_min


def pair_correlation(lon, lat, theta_bins):
    """Compute pair correlation function for points on the sphere.
    
    Parameters
    ----------
    lon : array-like
        Array of longitude coordinates (deg)
    lat : array-like
        Array of latitude coordinates (deg)
    theta_bins : array-like
        Array defining the `theta` binning (deg)
        `theta` is the angular offset between positions.
    Returns
    -------
    counts : array
        Array of point separations per `theta` bin.
    """
    # TODO: Implement speedups:
    # - use radians
    # - avoid processing each pair twice (distance a to b and b to a)
    counts = np.zeros(shape=len(theta_bins)-1, dtype=int)
    # If there are many points this should have acceptable performance
    # because the inner loop is in np.histogram, not in Python
    for ii in range(len(lon)):
        theta = separation(lon[ii], lat[ii], lon, lat)
        hist = np.histogram(theta, theta_bins)[0]
        counts += hist
    return counts

def sky_to_sky(lon, lat, in_system, out_system):
    """Convert between sky coordinates.
    
    lon : array-like
        Longitude coordinate array
    lat : array-like
        Latitude coordinate array
    in_system : {'galactic', 'icrs'}
        Input coordinate system
    out_system : {'galactic', 'icrs'}
        Output coordinate system
    """    
    from astropy.coordinates import ICRS, Galactic
    systems = dict(galactic=Galactic, icrs=ICRS)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    coords = systems[in_system](lon, lat, units='deg')
    return coords.transform_to(systems[out_system]).degrees
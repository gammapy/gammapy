# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Celestial coordinate utility functions.
"""
from __future__ import print_function, division
import numpy as np
from numpy import (cos, sin, arcsin,
                   arctan2, radians, degrees, pi)

__all__ = ['galactic_to_radec', 'radec_to_galactic', 'sky_to_sky']


def galactic_to_radec(glon, glat, unit='deg'):
    """Convert Galactic to Equatorial J2000 coordinates.

    Only accurate to ~ 3 digits.

    This is a standalone implementation that only uses `numpy`.
    Use it where you don't want to depend on a real celestial coordinate
    package like `astropy.coordinates` or `kapteyn.celestial`.

    Parameters
    ----------
    glon, glat : array_like
        Galactic coordinates
    unit : {'deg', 'rad'}
        Units of input and output coordinates

    Returns
    -------
    ra, dec : array_like
        Equatorial coordinates.
    """
    if unit == 'deg':
        glon, glat = radians(glon), radians(glat)

    ra_gp = radians(192.85948)
    de_gp = radians(27.12825)
    lcp = radians(122.932)

    term1 = cos(glat) * sin(lcp - glon)
    term2 = cos(de_gp) * sin(glat) - sin(de_gp) * cos(glat) * cos(lcp - glon)
    ramragp = arctan2(term1, term2)
    ra = (ramragp + ra_gp + 2 * pi) % (2 * pi)

    sin_d = sin(de_gp) * sin(glat) + cos(de_gp) * cos(glat) * cos(lcp - glon)
    dec = arcsin(sin_d)

    if unit == 'deg':
        ra, dec = degrees(ra), degrees(dec)    

    return ra, dec


def radec_to_galactic(ra, dec, unit='deg'):
    """Convert Equatorial J2000 to Galactic coordinates.

    Only accurate to ~ 3 digits.

    This is a standalone implementation that only uses `numpy`.
    Use it where you don't want to depend on a real celestial coordinate
    package like `astropy.coordinates` or `kapteyn.celestial`.

    Parameters
    ----------
    ra, dec : array_like
        Equatorial coordinates.
    unit : {'deg', 'rad'}
        Units of input and output coordinates

    Returns
    -------
    glon, glat : array_like
        Galactic coordinates
    """
    if unit == 'deg':
        ra, dec = radians(ra), radians(dec)

    ra_gp = radians(192.85948)
    de_gp = radians(27.12825)
    lcp = radians(122.932)

    term1 = cos(dec) * sin(ra - ra_gp)
    term2 = cos(de_gp) * sin(dec) - sin(de_gp) * cos(dec) * cos(ra - ra_gp)
    lcpml = arctan2(term1, term2)
    glon = (lcp - lcpml + 2 * pi) % (2 * pi)

    sin_b = sin(de_gp) * sin(dec) + cos(de_gp) * cos(dec) * cos(ra - ra_gp)
    glat = arcsin(sin_b)

    if unit == 'deg':
        glon, glat = degrees(glon), degrees(glat)

    return glon, glat


def sky_to_sky(lon, lat, in_system, out_system, unit='deg'):
    """Convert between sky coordinates.

    Parameters
    ----------
    lon, lat : array_like
        Coordinate arrays
    in_system, out_system : {'galactic', 'icrs'}
        Input / output coordinate system
    unit : {'deg', 'rad'}
        Units of input and output coordinates

    Returns
    -------
    """    
    from astropy.coordinates import ICRS, Galactic
    systems = dict(galactic=Galactic, icrs=ICRS)

    lon = np.asanyarray(lon)
    lat = np.asanyarray(lat)

    in_coords = systems[in_system](lon, lat, unit=(unit, unit))
    out_coords = in_coords.transform_to(systems[out_system])
    
    if unit == 'deg':
        return out_coords.lonangle.deg, out_coords.latangle.deg
    else:
        return out_coords.lonangle.rad, out_coords.latangle.rad

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Other coordinate and distance-related functions"""
from __future__ import print_function, division
from numpy import (cos, sin, arcsin, sqrt,
                   tan, arctan, arctan2,
                   radians, degrees, pi)
from astropy.constants import R_sun
from astropy.units import pc, kpc, cm, km, second, year

__all__ = ['cartesian', 'galactic', 'luminosity_to_flux', 'flux_to_luminosity',
           'radius_to_angle', 'angle_to_radius', 'spherical_velocity', 'motion_since_birth']


def cartesian(r, theta, dx=0, dy=0):
    """Takes polar coordinates r and theta and returns cartesian coordinates x and y.
    Has the option to add dx and dy for blurring the positions."""
    x = r * cos(theta) + dx
    y = r * sin(theta) + dy
    return x, y


def galactic(x, y, z):
    """Compute galactic coordinates lon, lat (deg) and distance (kpc)
    for given position in cartesian coordinates (kpc)"""
    d = sqrt(x ** 2 + (y - R_sun) ** 2 + z ** 2)
    lon = degrees(arctan2(x, R_sun - y))
    lat = degrees(arcsin(z / d))
    return lon, lat, d


def luminosity_to_flux(luminosity, distance):
    """Distance is assumed to be in kpc"""
    return luminosity / (4 * pi * (kpc.to(cm) * distance) ** 2)


def flux_to_luminosity(flux, distance):
    """Distance is assumed to be in kpc"""
    return flux * (4 * pi * (kpc.to(cm) * distance) ** 2)


def radius_to_angle(radius, distance):
    """Radius (pc), distance(kpc), angle(deg)"""
    return degrees(arctan(kpc.to(pc) * radius / distance))


def angle_to_radius(angle, distance):
    """Radius (pc), distance(kpc), angle(deg)"""
    return tan(radians(angle)) * kpc.to(pc) * distance


def spherical_velocity(x, y, z, vx, vy, vz):
    """Computes the projected angular velocity in spherical coordinates."""
    d = sqrt(x ** 2 + y ** 2 + z ** 2)
    r = sqrt(x ** 2 + y ** 2)
    
    v_lon = degrees(1. / (kpc.to(km) * r) * (-y * vx + x * vy)) * year.to(second) * 1e6
    v_lat = (degrees(vz / (sqrt(1 - (z / d) ** 2) * kpc.to(km) * d ) - 
              sqrt(vx ** 2 + vy ** 2 + vz ** 2) * z / 
              (kpc.to(km) * (sqrt(1 - (z / d) ** 2) * d ** 2))) * year.to(second) * 1e6)
    return v_lon, v_lat


def motion_since_birth(x, y, z, v, age, theta, phi):
    """Takes x[kpc], y[kpc], z[kpc], v[km/s] and age[years] chooses an arbitrary direction
    and computes the new position. Doesn't include any galactic potential modelation.
    Returns the new position."""
    vx = v * cos(phi) * sin(theta)
    vy = v * sin(phi) * sin(theta)
    vz = v * cos(theta)

    age = year.to(second) * age

    # Compute new positions
    x = x + kpc.to(km) * vx * age
    y = y + kpc.to(km) * vy * age
    z = z + kpc.to(km) * vz * age

    return x, y, z, vx, vy, vz

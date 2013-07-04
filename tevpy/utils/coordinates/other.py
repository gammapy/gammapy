# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Other coordinate and distance-related functions"""
from numpy import (cos, sin, arcsin, sqrt,
                   tan, arctan, arctan2,
                   radians, degrees, pi)
from const import (r_sun, sec_per_year, km_per_kpc,
                   cm_per_kpc, pc_per_kpc)


def cartesian(r, theta, dx=0, dy=0):
    """Takes polar coordinates r and theta and returns cartesian coordinates x and y.
    Has the option to add dx and dy for blurring the positions."""
    x = r * cos(theta) + dx
    y = r * sin(theta) + dy
    return x, y


def spherical(x, y, z):
    """Compute spherical coordinates lon, lat (deg) and distance (kpc)
    for given position in cartesian coordinates (kpc)"""
    d = sqrt(x ** 2 + (y - r_sun) ** 2 + z ** 2)
    lon = degrees(arctan2(x, r_sun - y))
    lat = degrees(arcsin(z / d))
    return lon, lat, d


def luminosity_to_flux(luminosity, distance):
    """Distance is assumed to be in kpc"""
    return luminosity / (4 * pi * (distance * cm_per_kpc) ** 2)


def flux_to_luminosity(flux, distance):
    """Distance is assumed to be in kpc"""
    return flux * (4 * pi * (distance * cm_per_kpc) ** 2)


def radius_to_angle(radius, distance):
    """Radius (pc), distance(kpc), angle(deg)"""
    return degrees(arctan(radius * pc_per_kpc / distance))


def angle_to_radius(angle, distance):
    """Radius (pc), distance(kpc), angle(deg)"""
    return tan(radians(angle)) * distance * pc_per_kpc


def spherical_velocity(x, y, z, vx, vy, vz):
    """Computes the projected angular velocity in spherical coordinates."""
    d = sqrt(x ** 2 + y ** 2 + z ** 2)
    r = sqrt(x ** 2 + y ** 2)
    v_lon = degrees(1. / (r * km_per_kpc) * (-y * vx + x * vy)) * sec_per_year * 1e6
    v_lat = (degrees(vz / (sqrt(1 - (z / d) ** 2) * d * km_per_kpc) - 
              sqrt(vx ** 2 + vy ** 2 + vz ** 2) * z / 
              (km_per_kpc * (sqrt(1 - (z / d) ** 2) * d ** 2))) * sec_per_year * 1e6)
    return v_lon, v_lat


def motion_since_birth(x, y, z, v, age, theta, phi):
    """Takes x[kpc], y[kpc], z[kpc], v[km/s] and age[years] chooses an arbitrary direction
    and computes the new position. Doesn't include any galactic potential modelation.
    Returns the new position."""
    vx = v * cos(phi) * sin(theta)
    vy = v * sin(phi) * sin(theta)
    vz = v * cos(theta)

    # Compute new positions
    x = x + vx * age * sec_per_year / km_per_kpc
    y = y + vy * age * sec_per_year / km_per_kpc
    z = z + vz * age * sec_per_year / km_per_kpc
    return x, y, z, vx, vy, vz

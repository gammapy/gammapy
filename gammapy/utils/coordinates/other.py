# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Other coordinate and distance-related functions"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Unit, Quantity

__all__ = [
    "cartesian",
    "galactic",
    "luminosity_to_flux",
    "flux_to_luminosity",
    "radius_to_angle",
    "angle_to_radius",
    "velocity_glon_glat",
    "motion_since_birth",
    "polar",
    "D_SUN_TO_GALACTIC_CENTER",
]

# TODO: replace this with the default from the Galactocentric frame in astropy.coordinates
D_SUN_TO_GALACTIC_CENTER = Quantity(8.5, "kpc")
"""Default assumed distance from the Sun to the Galactic center (`~astropy.units.Quantity`)"""


def cartesian(r, theta):
    """
    Convert polar coordinates to cartesian coordinates.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def polar(x, y):
    """Convert cartesian coordinates to polar coordinates."""
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def galactic(x, y, z, obs_pos=None):
    """Compute galactic coordinates lon, lat (deg) and distance (kpc)
    for given position in cartesian coordinates (kpc)"""
    obs_pos = obs_pos or [D_SUN_TO_GALACTIC_CENTER, 0, 0]
    y_prime = y + D_SUN_TO_GALACTIC_CENTER
    d = np.sqrt(x ** 2 + y_prime ** 2 + z ** 2)
    glon = np.arctan2(x, y_prime).to("deg")
    glat = np.arcsin(z / d).to("deg")
    return d, glon, glat


def luminosity_to_flux(luminosity, distance):
    """Distance is assumed to be in kpc"""
    return luminosity / (4 * np.pi * distance ** 2)


def flux_to_luminosity(flux, distance):
    """Distance is assumed to be in kpc"""
    return flux * 4 * np.pi * distance ** 2


def radius_to_angle(radius, distance):
    """Radius (pc), distance(kpc), angle(deg)"""
    return np.arctan(radius / distance)


def angle_to_radius(angle, distance):
    """Radius (pc), distance(kpc), angle(deg)"""
    return np.tan(angle * distance)


def velocity_glon_glat(x, y, z, vx, vy, vz):
    """
    Compute projected angular velocity in galactic coordinates.

    Parameters
    ----------
    x : `~astropy.units.Quantity`
        Position in x direction
    y : `~astropy.units.Quantity`
        Position in y direction
    z : `~astropy.units.Quantity`
        Position in z direction
    vx : `~astropy.units.Quantity`
        Velocity in x direction
    vy : `~astropy.units.Quantity`
        Velocity in y direction
    vz : `~astropy.units.Quantity`
        Velocity in z direction

    Returns
    -------
    v_glon : `~astropy.units.Quantity`
        Projected velocity in Galactic longitude
    v_glat : `~astropy.units.Quantity`
        Projected velocity in Galactic latitude
    """
    y_prime = y + D_SUN_TO_GALACTIC_CENTER
    d = np.sqrt(x ** 2 + y_prime ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y_prime ** 2)

    v_glon = (-y_prime * vx + x * vy) / r ** 2
    v_glat = vz / (np.sqrt(1 - (z / d) ** 2) * d) - np.sqrt(
        vx ** 2 + vy ** 2 + vz ** 2
    ) * z / ((np.sqrt(1 - (z / d) ** 2) * d ** 2))
    return v_glon * Unit("rad"), v_glat * Unit("rad")


def motion_since_birth(v, age, theta, phi):
    """
    Compute motion of a object with given velocity, direction and age.

    Parameters
    ----------
    v : `~astropy.units.Quantity`
        Absolute value of the velocity
    age : `~astropy.units.Quantity`
        Age of the source.
    theta : `~astropy.units.Quantity`
        Angular direction of the velocity.
    phi : `~astropy.units.Quantity`
        Angular direction of the velocity.

    Returns
    -------
    dx : `~astropy.units.Quantity`
        Displacement in x direction
    dy : `~astropy.units.Quantity`
        Displacement in y direction
    dz : `~astropy.units.Quantity`
        Displacement in z direction
    vx : `~astropy.units.Quantity`
        Velocity in x direction
    vy : `~astropy.units.Quantity`
        Velocity in y direction
    vz : `~astropy.units.Quantity`
        Velocity in z direction
    """
    vx = v * np.cos(phi) * np.sin(theta)
    vy = v * np.sin(phi) * np.sin(theta)
    vz = v * np.cos(theta)

    # Compute new positions
    dx = vx * age
    dy = vy * age
    dz = vz * age
    return dx, dy, dz, vx, vy, vz

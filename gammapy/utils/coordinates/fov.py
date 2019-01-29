# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame

__all__ = ["fov_to_sky", "sky_to_fov"]

def fov_to_sky(lon, lat, lon_pnt, lat_pnt):
    """Make a transformation from field-of-view coordinates to sky coordinates.

    Parameters
    ----------
    lon : `~astropy.units.Quantity`
        Field-of-view longitude coordinate to be transformed
    lat : `~astropy.units.Quantity`
        Field-of-view latitude coordinate to be transformed
    lon_pnt : `~astropy.units.Quantity`
        Longitude coordinate of the pointing position
    lat_pnt : `~astropy.units.Quantity`
        Latitude coordinate of the pointing position

    Returns
    -------
    lon_t : `~astropy.units.Quantity`
        Sky longitude coordinate
    lat_t : `~astropy.units.Quantity`
        Sky latitude coordinate
    """

    # Create a frame that is centered on the pointing position
    center = SkyCoord(lon_pnt, lat_pnt)
    fov_frame = SkyOffsetFrame(origin=center)

    # Define coordinate to be transformed.
    # Need to switch the sign of the longitude angle here
    # because this axis is reversed in our definition of the FoV-system
    target_fov = SkyCoord(-lon, lat, frame=fov_frame)

    # Transform into celestial system (need not be ICRS)
    target_sky = target_fov.icrs

    return target_sky.ra, target_sky.dec


def sky_to_fov(lon, lat, lon_pnt, lat_pnt):
    """Make a transformation from sky coordinates to field-of-view coordinates.

    Parameters
    ----------
    lon : `~astropy.units.Quantity`
        Sky longitude coordinate to be transformed
    lat : `~astropy.units.Quantity`
        Sky latitude coordinate to be transformed
    lon_pnt : `~astropy.units.Quantity`
        Longitude coordinate of the pointing position
    lat_pnt : `~astropy.units.Quantity`
        Latitude coordinate of the pointing position

    Returns
    -------
    lon_t : `~astropy.units.Quantity`
        Field-of-view longitude coordinate
    lat_t : `~astropy.units.Quantity`
        Field-of-view latitude coordinate
    """

    # Create a frame that is centered on the pointing position
    center = SkyCoord(lon_pnt, lat_pnt)
    fov_frame = SkyOffsetFrame(origin=center)

    # Define coordinate to be transformed.
    target_sky = SkyCoord(lon, lat)

    # Transform into FoV-system
    target_fov = target_sky.transform_to(fov_frame)

    # Switch sign of longitude angle since this axis is
    # reversed in our definition of the FoV-system
    return -target_fov.lon, target_fov.lat

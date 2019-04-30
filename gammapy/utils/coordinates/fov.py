# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.coordinates import SkyCoord, SkyOffsetFrame

__all__ = ["fov_to_sky", "sky_to_fov"]


def fov_to_sky(lon, lat, lon_pnt, lat_pnt):
    """Transform field-of-view coordinates to sky coordinates.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Field-of-view coordinate to be transformed
    lon_pnt, lat_pnt : `~astropy.units.Quantity`
        Coordinate specifying the pointing position
        (i.e. the center of the field of view)

    Returns
    -------
    lon_t, lat_t : `~astropy.units.Quantity`
        Transformed sky coordinate
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
    """Transform sky coordinates to field-of-view coordinates.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Sky coordinate to be transformed
    lon_pnt, lat_pnt : `~astropy.units.Quantity`
        Coordinate specifying the pointing position
        (i.e. the center of the field of view)

    Returns
    -------
    lon_t, lat_t : `~astropy.units.Quantity`
        Transformed field-of-view coordinate
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

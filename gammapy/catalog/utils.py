# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog utility functions / classes."""
import numpy as np
from astropy.coordinates import Angle, SkyCoord

__all__ = ["skycoord_from_table", "select_sky_box", "select_sky_circle"]


def skycoord_from_table(table):
    """Make `~astropy.coordinates.SkyCoord` from lon, lat columns in `~astropy.table.Table`.

    This is a convenience function similar to `~astropy.coordinates.SkyCoord.guess_from_table`,
    but with the column names we usually use.

    TODO: I'm not sure if it's a good idea to use this because it's not always clear
    which positions are taken.
    """
    try:
        keys = table.colnames
    except AttributeError:
        keys = table.keys()

    if {"RAJ2000", "DEJ2000"}.issubset(keys):
        lon, lat, frame = "RAJ2000", "DEJ2000", "icrs"
    elif {"RA", "DEC"}.issubset(keys):
        lon, lat, frame = "RA", "DEC", "icrs"
    elif {"GLON", "GLAT"}.issubset(keys):
        lon, lat, frame = "GLON", "GLAT", "galactic"
    elif {"glon", "glat"}.issubset(keys):
        lon, lat, frame = "glon", "glat", "galactic"
    else:
        raise KeyError("No column GLON / GLAT or RA / DEC or RAJ2000 / DEJ2000 found.")

    unit = table[lon].unit.to_string() if table[lon].unit else "deg"
    skycoord = SkyCoord(table[lon], table[lat], unit=unit, frame=frame)

    return skycoord


def select_sky_box(table, lon_lim, lat_lim, frame="icrs", inverted=False):
    """Select sky positions in a box.

    This function can be applied e.g. to event lists of source catalogs
    or observation tables.

    Note: if useful we can add a function that returns the mask
    or indices instead of applying the selection directly

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with sky coordinate columns.
    lon_lim, lat_lim : `~astropy.coordinates.Angle`
        Box limits (each should be a min, max tuple).
    frame : str, optional
        Frame in which to apply the box cut.
        Built-in Astropy coordinate frames are supported, e.g.
        'icrs', 'fk5' or 'galactic'.
    inverted : bool, optional
        Invert selection: keep all entries outside the selected region.

    Returns
    -------
    table : `~astropy.table.Table`
        Copy of input table with box cut applied.

    Examples
    --------
    >>> selected_obs_table = select_sky_box(obs_table,
    ...                                     lon_lim=Angle([150, 300], 'deg'),
    ...                                     lat_lim=Angle([-50, 0], 'deg'),
    ...                                     frame='icrs')
    """
    skycoord = skycoord_from_table(table)
    skycoord = skycoord.transform_to(frame)
    lon = skycoord.data.lon
    lat = skycoord.data.lat
    # SkyCoord automatically wraps lon angles at 360 deg, so in case
    # the lon range is wrapped at 180 deg, lon angles must be wrapped
    # also at 180 deg for the comparison to work
    if any(l < Angle(0.0, "deg") for l in lon_lim):
        lon = lon.wrap_at(Angle(180, "deg"))

    lon_mask = (lon_lim[0] <= lon) & (lon < lon_lim[1])
    lat_mask = (lat_lim[0] <= lat) & (lat < lat_lim[1])
    mask = lon_mask & lat_mask
    if inverted:
        mask = np.invert(mask)

    return table[mask]


def select_sky_circle(table, lon_cen, lat_cen, radius, frame="icrs", inverted=False):
    """Select sky positions in a circle.

    This function can be applied e.g. to event lists of source catalogs
    or observation tables.

    Note: if useful we can add a function that returns the mask
    or indices instead of applying the selection directly

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with sky coordinate columns.
    lon_cen, lat_cen : `~astropy.coordinates.Angle`
        Circle center.
    radius : `~astropy.coordinates.Angle`
        Circle radius.
    frame : str, optional
        Frame in which to apply the box cut.
        Built-in Astropy coordinate frames are supported, e.g.
        'icrs', 'fk5' or 'galactic'.
    inverted : bool, optional
        Invert selection: keep all entries outside the selected region.

    Returns
    -------
    table : `~astropy.table.Table`
        Copy of input table with circle cut applied.

    Examples
    --------
    >>> selected_obs_table = select_sky_circle(obs_table,
    ...                                        lon=Angle(0, 'deg'),
    ...                                        lat=Angle(0, 'deg'),
    ...                                        radius=Angle(5, 'deg'),
    ...                                        frame='galactic')
    """
    skycoord = skycoord_from_table(table)
    skycoord = skycoord.transform_to(frame)
    # no need to wrap lon angleshere, since the SkyCoord separation
    # method takes care of it
    center = SkyCoord(lon_cen, lat_cen, frame=frame)
    ang_distance = skycoord.separation(center)

    mask = ang_distance < radius
    if inverted:
        mask = np.invert(mask)

    return table[mask]

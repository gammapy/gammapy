# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = ["fov_to_sky", "sky_to_fov"]


def fov_to_sky(lon, lat, lon_pnt, lat_pnt):
    """Make a transformation from field-of-view coordinates to sky coordinates.

    Parameters
    ----------
    lon : array_like
        Field-of-view longitude coordinate to be transformed, in degrees
    lat : array_like
        Field-of-view latitude coordinate to be transformed, in degrees
    lon_pnt : array_like
        Longitude coordinate of the pointing position, in degrees
    lat_pnt : array_like
        Latitude coordinate of the pointing position, in degrees

    Returns
    -------
    lon_t : array_like
        Sky longitude coordinate, in degrees
    lat_t : array_like
        Sky latitude coordinate, in degrees
    """

    # compute cartesian coordinates
    x = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = np.sin(np.deg2rad(lat))

    # make sure vector is properly normalised
    assert np.allclose(x ** 2 + y ** 2 + z ** 2, 1)

    # switch coordinates due to axis convention
    x_ = -z
    y_ = y
    z_ = x

    # transform
    lat_pnt = np.deg2rad(90 - lat_pnt)
    lon_pnt = -np.deg2rad(lon_pnt)
    x_t = (
        x_ * np.cos(lat_pnt) * np.cos(lon_pnt)
        - y_ * np.sin(lon_pnt)
        + z_ * np.sin(lat_pnt) * np.cos(lon_pnt)
    )
    y_t = (
        x_ * np.sin(lon_pnt) * np.cos(lat_pnt)
        + y_ * np.cos(lon_pnt)
        + z_ * np.sin(lon_pnt) * np.sin(lat_pnt)
    )
    z_t = -x_ * np.sin(lat_pnt) + z_ * np.cos(lat_pnt)

    # compute new lon, lat
    lon_t = -np.rad2deg(np.arctan2(y_t, x_t))
    lat_t = np.rad2deg(np.arcsin(z_t))

    # shift lon by 360 degrees if negative
    lon_t = np.where(lon_t < 0, lon_t + 360, lon_t)

    return lon_t, lat_t


def sky_to_fov(lon, lat, lon_pnt, lat_pnt):
    """Make a transformation from sky coordinates to field-of-view coordinates.

    Parameters
    ----------
    lon : array_like
        Sky longitude coordinate to be transformed, in degrees
    lat : array_like
        Sky latitude coordinate to be transformed, in degrees
    lon_pnt : array_like
        Longitude coordinate of the pointing position, in degrees
    lat_pnt : array_like
        Latitude coordinate of the pointing position, in degrees

    Returns
    -------
    lon_t : array_like
        Field-of-view longitude coordinate, in degrees
    lat_t : array_like
        Field-of-view latitude coordinate, in degrees
    """

    lon_ = -lon

    # compute cartesian coordinates
    x = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon_))
    y = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon_))
    z = np.sin(np.deg2rad(lat))

    # make sure vector is properly normalised
    assert np.allclose(x ** 2 + y ** 2 + z ** 2, 1)

    # transform
    lat_pnt = np.deg2rad(90 - lat_pnt)
    lon_pnt = -np.deg2rad(lon_pnt)
    x_t = (
        x * np.cos(lat_pnt) * np.cos(lon_pnt)
        + y * np.sin(lon_pnt) * np.cos(lat_pnt)
        - z * np.sin(lat_pnt)
    )
    y_t = -x * np.sin(lon_pnt) + y * np.cos(lon_pnt)
    z_t = (
        x * np.sin(lat_pnt) * np.cos(lon_pnt)
        + y * np.sin(lat_pnt) * np.sin(lon_pnt)
        + z * np.cos(lat_pnt)
    )

    # switch coordinates due to axis convention
    x_ = z_t
    y_ = y_t
    z_ = -x_t

    # compute new lon, lat
    lon_t = np.rad2deg(np.arctan2(y_, x_))
    lat_t = np.rad2deg(np.arcsin(z_))

    return lon_t, lat_t

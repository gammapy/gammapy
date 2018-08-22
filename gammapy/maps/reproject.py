# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u

__all__ = ["reproject_car_to_hpx", "reproject_car_to_wcs"]


def _get_input_pix_celestial(wcs_in, wcs_out, shape_out):
    """
    Get the pixel coordinates of the pixels in an array of shape ``shape_out``
    in the input WCS.
    """
    from reproject.wcs_utils import convert_world_coordinates

    # TODO: for now assuming that coordinates are spherical, not
    # necessarily the case. Also assuming something about the order of the
    # arguments.

    # Generate pixel coordinates of output image
    xp_out, yp_out = np.indices(shape_out, dtype=float)[::-1]

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    xw_out, yw_out = wcs_out.wcs_pix2world(xp_out, yp_out, 0)

    xw_in, yw_in = convert_world_coordinates(xw_out, yw_out, wcs_out, wcs_in)

    xp_in, yp_in = wcs_in.wcs_world2pix(xw_in, yw_in, 0)

    return xp_in, yp_in


def reproject_car_to_hpx(input_data, coord_system_out, nside, order=1, nested=False):
    import healpy as hp
    from scipy.ndimage import map_coordinates
    from reproject.wcs_utils import convert_world_coordinates
    from reproject.healpix.utils import parse_coord_system

    data, wcs_in = input_data

    npix = hp.nside2npix(nside)

    theta, phi = hp.pix2ang(nside, np.arange(npix), nested)
    lon_out = np.degrees(phi)
    lat_out = 90. - np.degrees(theta)

    # Convert between celestial coordinates
    coord_system_out = parse_coord_system(coord_system_out)
    with np.errstate(invalid="ignore"):
        lon_in, lat_in = convert_world_coordinates(
            lon_out, lat_out, (coord_system_out, u.deg, u.deg), wcs_in
        )

    # Look up pixels in input system
    yinds, xinds = wcs_in.wcs_world2pix(lon_in, lat_in, 0)

    # Interpolate
    data = np.pad(data, 3, mode="wrap")

    healpix_data = map_coordinates(
        data, [xinds + 3, yinds + 3], order=order, mode="wrap", cval=np.nan
    )

    return healpix_data, (~np.isnan(healpix_data)).astype(float)


def reproject_car_to_wcs(input_data, wcs_out, shape_out, order=1):
    """Reproject an all-sky CAR projection to another WCS projection.

    This method performs special handling of the projection edges to
    ensure that the interpolation of the CAR projection is correctly
    wrapped in longitude.
    """
    from scipy.ndimage import map_coordinates

    slice_in, wcs_in = input_data

    array_new = np.zeros(shape_out)
    slice_out = array_new

    xp_in, yp_in = _get_input_pix_celestial(
        wcs_in.celestial, wcs_out.celestial, slice_out.shape
    )
    coordinates = np.array([yp_in.ravel(), xp_in.ravel()])

    jmin, imin = np.floor(np.nanmin(coordinates, axis=1)).astype(int) - 1
    jmax, imax = np.ceil(np.nanmax(coordinates, axis=1)).astype(int) + 1

    ny, nx = slice_in.shape

    if imin >= nx or imax < 0 or jmin >= ny or jmax < 0:
        return array_new * np.nan, array_new.astype(float)

    # Pad by 3 pixels to ensure that cubic interpolation works
    slice_in = np.pad(slice_in, 3, mode="wrap")

    # Make sure image is floating point. We do this only now because
    # we want to avoid converting the whole input array if possible
    slice_in = np.asarray(slice_in, dtype=float)
    slice_out[:, :] = map_coordinates(
        slice_in, coordinates + 3, order=order, cval=np.nan, mode="constant"
    ).reshape(slice_out.shape)

    return array_new, (~np.isnan(array_new)).astype(float)

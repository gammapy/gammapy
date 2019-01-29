# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from ..fov import fov_to_sky, sky_to_fov
from astropy.units import deg


def test_fov_to_sky():
    # test some simple cases
    az, alt = fov_to_sky(1*deg, 1*deg, 0*deg, 0*deg)
    assert_allclose(az.value, 359)
    assert_allclose(alt.value, 1)

    az, alt = fov_to_sky(-1*deg, 1*deg, 180*deg, 0*deg)
    assert_allclose(az.value, 181)
    assert_allclose(alt.value, 1)

    az, alt = fov_to_sky(1*deg, 0*deg, 0*deg, 60*deg)
    assert_allclose(az.value, 358, rtol=1e-3)
    assert_allclose(alt.value, 59.985, rtol=1e-3)

    # these are cross-checked with the
    # transformation as implemented in H.E.S.S.
    fov_altaz_lon = np.array([0.7145614, 0.86603433, -0.05409698, 2.10295248])
    fov_altaz_lat = np.array([-1.60829115, -1.19643974, 0.45800984, 3.26844192])
    az_pointing = np.array([52.42056255, 52.24706061, 52.06655505, 51.86795724])
    alt_pointing = np.array([51.11908203, 51.23454751, 51.35376141, 51.48385814])
    az, alt = fov_to_sky(fov_altaz_lon*deg, fov_altaz_lat*deg, az_pointing*deg, alt_pointing*deg)
    assert_allclose(az.value, np.array([51.320575, 50.899125, 52.154053, 48.233023]))
    assert_allclose(alt.value, np.array([49.505451, 50.030165, 51.811739, 54.700102]))


def test_sky_to_fov():
    # test some simple cases
    lon, lat = sky_to_fov(1*deg, 1*deg, 0*deg, 0*deg)
    assert_allclose(lon.value, -1)
    assert_allclose(lat.value, 1)

    lon, lat = sky_to_fov(269*deg, 0*deg, 270*deg, 0*deg)
    assert_allclose(lon.value, 1)
    assert_allclose(lat.value, 0, atol=1e-7)

    lon, lat = sky_to_fov(1*deg, 60*deg, 0*deg, 60*deg)
    assert_allclose(lon.value, -0.5, rtol=1e-3)
    assert_allclose(lat.value, 0.003779, rtol=1e-3)

    # these are cross-checked with the
    # transformation as implemented in H.E.S.S.
    az = np.array([51.320575, 50.899125, 52.154053, 48.233023])
    alt = np.array([49.505451, 50.030165, 51.811739, 54.700102])
    az_pointing = np.array([52.42056255, 52.24706061, 52.06655505, 51.86795724])
    alt_pointing = np.array([51.11908203, 51.23454751, 51.35376141, 51.48385814])
    lon, lat = sky_to_fov(az*deg, alt*deg, az_pointing*deg, alt_pointing*deg)
    assert_allclose(
        lon.value, np.array([0.7145614, 0.86603433, -0.05409698, 2.10295248]), rtol=1e-5
    )
    assert_allclose(
        lat.value, np.array([-1.60829115, -1.19643974, 0.45800984, 3.26844192]), rtol=1e-5
    )


# def test_fov_to_sky():
#     # test some simple cases
#     az, alt = fov_to_sky(1, 1, 0, 0)
#     assert_allclose(az, 359)
#     assert_allclose(alt, 1)

#     az, alt = fov_to_sky(-1, 1, 180, 0)
#     assert_allclose(az, 181)
#     assert_allclose(alt, 1)

#     az, alt = fov_to_sky(1, 0, 0, 60)
#     assert_allclose(az, 358, rtol=1e-3)
#     assert_allclose(alt, 59.985, rtol=1e-3)

#     # these are cross-checked with the
#     # transformation as implemented in H.E.S.S.
#     fov_altaz_lon = np.array([0.7145614, 0.86603433, -0.05409698, 2.10295248])
#     fov_altaz_lat = np.array([-1.60829115, -1.19643974, 0.45800984, 3.26844192])
#     az_pointing = np.array([52.42056255, 52.24706061, 52.06655505, 51.86795724])
#     alt_pointing = np.array([51.11908203, 51.23454751, 51.35376141, 51.48385814])
#     az, alt = fov_to_sky(fov_altaz_lon, fov_altaz_lat, az_pointing, alt_pointing)
#     assert_allclose(az, np.array([51.320575, 50.899125, 52.154053, 48.233023]))
#     assert_allclose(alt, np.array([49.505451, 50.030165, 51.811739, 54.700102]))


# def test_sky_to_fov():
#     # test some simple cases
#     lon, lat = sky_to_fov(1, 1, 0, 0)
#     assert_allclose(lon, -1)
#     assert_allclose(lat, 1)

#     lon, lat = sky_to_fov(269, 0, 270, 0)
#     assert_allclose(lon, 1)
#     assert_allclose(lat, 0, atol=1e-7)

#     lon, lat = sky_to_fov(1, 60, 0, 60)
#     assert_allclose(lon, -0.5, rtol=1e-3)
#     assert_allclose(lat, 0.003779, rtol=1e-3)

#     # these are cross-checked with the
#     # transformation as implemented in H.E.S.S.
#     az = np.array([51.320575, 50.899125, 52.154053, 48.233023])
#     alt = np.array([49.505451, 50.030165, 51.811739, 54.700102])
#     az_pointing = np.array([52.42056255, 52.24706061, 52.06655505, 51.86795724])
#     alt_pointing = np.array([51.11908203, 51.23454751, 51.35376141, 51.48385814])
#     lon, lat = sky_to_fov(az, alt, az_pointing, alt_pointing)
#     assert_allclose(
#         lon, np.array([0.7145614, 0.86603433, -0.05409698, 2.10295248]), rtol=1e-5
#     )
#     assert_allclose(
#         lat, np.array([-1.60829115, -1.19643974, 0.45800984, 3.26844192]), rtol=1e-5
#     )

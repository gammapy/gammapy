# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.utils.coordinates import fov_to_sky, sky_to_fov


def test_fov_to_sky():
    # test some simple cases
    az, alt = fov_to_sky(1 * u.deg, 1 * u.deg, 0 * u.deg, 0 * u.deg)
    assert_allclose(az.value, 359)
    assert_allclose(alt.value, 1)

    az, alt = fov_to_sky(-1 * u.deg, 1 * u.deg, 180 * u.deg, 0 * u.deg)
    assert_allclose(az.value, 181)
    assert_allclose(alt.value, 1)

    az, alt = fov_to_sky(1 * u.deg, 0 * u.deg, 0 * u.deg, 60 * u.deg)
    assert_allclose(az.value, 358, rtol=1e-3)
    assert_allclose(alt.value, 59.985, rtol=1e-3)

    # these are cross-checked with the
    # transformation as implemented in H.E.S.S.
    fov_altaz_lon = [0.7145614, 0.86603433, -0.05409698, 2.10295248]
    fov_altaz_lat = [-1.60829115, -1.19643974, 0.45800984, 3.26844192]
    az_pointing = [52.42056255, 52.24706061, 52.06655505, 51.86795724]
    alt_pointing = [51.11908203, 51.23454751, 51.35376141, 51.48385814]
    az, alt = fov_to_sky(
        fov_altaz_lon * u.deg,
        fov_altaz_lat * u.deg,
        az_pointing * u.deg,
        alt_pointing * u.deg,
    )
    assert_allclose(az.value, [51.320575, 50.899125, 52.154053, 48.233023])
    assert_allclose(alt.value, [49.505451, 50.030165, 51.811739, 54.700102])


def test_sky_to_fov():
    # test some simple cases
    lon, lat = sky_to_fov(1 * u.deg, 1 * u.deg, 0 * u.deg, 0 * u.deg)
    assert_allclose(lon.value, -1)
    assert_allclose(lat.value, 1)

    lon, lat = sky_to_fov(269 * u.deg, 0 * u.deg, 270 * u.deg, 0 * u.deg)
    assert_allclose(lon.value, 1)
    assert_allclose(lat.value, 0, atol=1e-7)

    lon, lat = sky_to_fov(1 * u.deg, 60 * u.deg, 0 * u.deg, 60 * u.deg)
    assert_allclose(lon.value, -0.5, rtol=1e-3)
    assert_allclose(lat.value, 0.003779, rtol=1e-3)

    # these are cross-checked with the
    # transformation as implemented in H.E.S.S.
    az = [51.320575, 50.899125, 52.154053, 48.233023]
    alt = [49.505451, 50.030165, 51.811739, 54.700102]
    az_pointing = [52.42056255, 52.24706061, 52.06655505, 51.86795724]
    alt_pointing = [51.11908203, 51.23454751, 51.35376141, 51.48385814]
    lon, lat = sky_to_fov(
        az * u.deg, alt * u.deg, az_pointing * u.deg, alt_pointing * u.deg
    )
    assert_allclose(
        lon.value, [0.7145614, 0.86603433, -0.05409698, 2.10295248], rtol=1e-5
    )
    assert_allclose(
        lat.value, [-1.60829115, -1.19643974, 0.45800984, 3.26844192], rtol=1e-5
    )

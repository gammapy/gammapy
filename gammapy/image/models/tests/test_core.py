# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import numpy as np
import astropy.units as u
from ....maps import Map
from ....utils.testing import requires_data
from ..core import (
    SkyPointSource,
    SkyGaussian,
    SkyElongatedGaussian,
    SkyDisk,
    SkyShell,
    SkyDiffuseConstant,
    SkyDiffuseMap,
)


def test_sky_point_source():
    model = SkyPointSource(lon_0="2.5 deg", lat_0="2.5 deg")
    lat, lon = np.mgrid[0:6, 0:6] * u.deg
    val = model(lon, lat)
    assert val.unit == "deg-2"
    assert_allclose(val.sum().value, 1)


def test_sky_gaussian():
    model = SkyGaussian(lon_0="1 deg", lat_0="45 deg", sigma="1 deg")
    lon = [1, 359] * u.deg
    lat = 46 * u.deg
    val = model(lon, lat)
    assert val.unit == "deg-2"
    assert model.parameters["sigma"].min == 0
    assert_allclose(val.to_value("sr-1"), [316.8970202, 118.6505303])


def test_sky_elongated_gaussian():
    model = SkyElongatedGaussian(
        lon_0="1 deg",
        lat_0="10 deg",
        sigma_lon="1 deg",
        sigma_lat="0.5 deg",
        theta="0.5 rad",
    )
    lon = [1, 0.5, 359] * u.deg
    lat = 11 * u.deg
    val = model(lon, lat)
    assert val.unit == "deg-2"
    assert model.parameters["sigma_lon"].min == 0
    assert model.parameters["sigma_lat"].min == 0
    desired = [199.63632071, 303.81143696, 84.92913364]
    assert_allclose(val.to_value("sr-1"), desired)


def test_sky_disk():
    model = SkyDisk(lon_0="1 deg", lat_0="45 deg", r_0="2 deg")
    lon = [1, 5, 359] * u.deg
    lat = 46 * u.deg
    val = model(lon, lat)
    assert val.unit == "sr-1"
    desired = [261.263956, 0, 261.263956]
    assert_allclose(val.value, desired)


def test_sky_shell():
    model = SkyShell(lon_0="1 deg", lat_0="45 deg", radius="2 deg", width="2 deg")

    lon = [1, 2, 4] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == "deg-2"
    desired = [55.979449, 57.831651, 94.919895]
    assert_allclose(val.to_value("sr-1"), desired)


def test_sky_diffuse_constant():
    model = SkyDiffuseConstant(value="42 sr-1")
    lon = [1, 2] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == "sr-1"
    assert_allclose(val.value, 42)


@requires_data("gammapy-data")
def test_sky_diffuse_map():
    filename = "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/RXJ1713_2016_250GeV.fits"
    model = SkyDiffuseMap.read(filename, normalize=False)
    lon = [258.5, 0] * u.deg
    lat = -39.8 * u.deg
    val = model(lon, lat)
    assert val.unit == "sr-1"
    desired = [3269.178107, 0]
    assert_allclose(val.value, desired)


@requires_data("gammapy-data")
def test_sky_diffuse_map_normalize():
    # define model map with a constant value of 1
    model_map = Map.create(map_type="wcs", width=(10, 5), binsz=0.5)
    model_map.data += 1.0
    model = SkyDiffuseMap(model_map)

    # define data map with a different spatial binning
    data_map = Map.create(map_type="wcs", width=(10, 5), binsz=1)
    coords = data_map.geom.get_coord()
    solid_angle = data_map.geom.solid_angle()
    vals = model(coords.lon * u.deg, coords.lat * u.deg) * solid_angle

    assert vals.unit == ""
    integral = vals.sum()
    assert_allclose(integral.value, 1, rtol=1e-4)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import numpy as np
import astropy.units as u
from ....maps import Map, WcsGeom
from ....utils.testing import requires_data
from ..core import (
    SkyPointSource,
    SkyGaussian,
    SkyDisk,
    SkyEllipse,
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
    model = SkyGaussian(lon_0="5 deg", lat_0="15 deg", sigma="1 deg")
    assert model.parameters["sigma"].min == 0
    val_0 = model(5 * u.deg, 15 * u.deg)
    val_sigma = model(5 * u.deg, 16 * u.deg)
    assert val_0.unit == "sr-1"
    ratio = val_0 / val_sigma
    assert_allclose(ratio, np.exp(0.5))


def test_sky_disk():
    model = SkyDisk(lon_0="1 deg", lat_0="45 deg", r_0="2 deg")
    lon = [1, 5, 359] * u.deg
    lat = 46 * u.deg
    val = model(lon, lat)
    assert val.unit == "sr-1"
    desired = [261.263956, 0, 261.263956]
    assert_allclose(val.value, desired)


def test_sky_ellipse():
    model = SkyEllipse(2 * u.deg, 2 * u.deg, 1 * u.deg, 0.8, 30 * u.deg)
    m_allsky = WcsGeom.create(
        binsz=0.01, width=(3, 3), skydir=(2, 2), coordsys="GAL", proj="AIT"
    )
    coords = m_allsky.get_coord()
    lon = coords.lon * u.deg
    lat = coords.lat * u.deg
    vals = model(lon, lat)
    assert vals.unit == "sr-1"
    mymap = Map.from_geom(m_allsky, data=vals.value)
    assert_allclose(
        np.sum(mymap.quantity * u.sr ** -1 * m_allsky.solid_angle()), 1, rtol=5.0e-4
    )


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

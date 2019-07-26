# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
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
    radius = model.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, 0)
    assert model.frame == "galactic"

    assert_allclose(model.position.l.deg, 2.5)
    assert_allclose(model.position.b.deg, 2.5)


def test_sky_gaussian():
    sigma = 1 * u.deg
    model = SkyGaussian(lon_0="5 deg", lat_0="15 deg", sigma=sigma)
    assert model.parameters["sigma"].min == 0
    val_0 = model(5 * u.deg, 15 * u.deg)
    val_sigma = model(5 * u.deg, 16 * u.deg)
    assert val_0.unit == "sr-1"
    ratio = val_0 / val_sigma
    assert_allclose(ratio, np.exp(0.5))
    radius = model.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, 5 * sigma.value)


def test_sky_disk():
    r_0 = 2 * u.deg
    model = SkyDisk(lon_0="1 deg", lat_0="45 deg", r_0=r_0)
    lon = [1, 5, 359] * u.deg
    lat = 46 * u.deg
    val = model(lon, lat)
    assert val.unit == "sr-1"
    desired = [261.263956, 0, 261.263956]
    assert_allclose(val.value, desired)
    radius = model.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, r_0.value)


def test_sky_disk_edge():
    r_0 = 2 * u.deg
    model = SkyDisk(lon_0="0 deg", lat_0="0 deg", r_0=r_0)
    value_center = model(0 * u.deg, 0 * u.deg)
    value_edge = model(r_0, 0 * u.deg)
    assert_allclose((value_edge / value_center).to_value(""), 0.5)

    edge = model.edge.quantity
    value_edge_pwidth = model(r_0 + edge / 2, 0 * u.deg)
    assert_allclose((value_edge_pwidth / value_center).to_value(""), 0.05)

    value_edge_nwidth = model(r_0 - edge / 2, 0 * u.deg)
    assert_allclose((value_edge_nwidth / value_center).to_value(""), 0.95)


def test_sky_ellipse():
    pytest.importorskip("astropy", minversion="3.1.1")
    # test the normalization for an elongated ellipse near the Galactic Plane
    m_geom_1 = WcsGeom.create(
        binsz=0.015, width=(20, 20), skydir=(2, 2), coordsys="GAL", proj="AIT"
    )
    coords = m_geom_1.get_coord()
    lon = coords.lon * u.deg
    lat = coords.lat * u.deg
    semi_major = 10 * u.deg
    model_1 = SkyEllipse(2 * u.deg, 2 * u.deg, semi_major, 0.4, 30 * u.deg)
    vals_1 = model_1(lon, lat)
    assert vals_1.unit == "sr-1"
    mymap_1 = Map.from_geom(m_geom_1, data=vals_1.value)
    assert_allclose(
        np.sum(mymap_1.quantity * u.sr ** -1 * m_geom_1.solid_angle()), 1, rtol=1.0e-3
    )

    radius = model_1.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, semi_major.value)
    # test rotation
    semi_major = 2 * u.deg
    semi_minor = 1 * u.deg
    eccentricity = np.sqrt(1 - (semi_minor / semi_major) ** 2)
    model_rot_test = SkyEllipse(
        0 * u.deg, 0 * u.deg, semi_major, eccentricity, 90 * u.deg
    )
    assert_allclose(model_rot_test(0 * u.deg, 1.5 * u.deg).value, 0)

    # test the normalization for a disk (ellipse with e=0) at the Galactic Pole,
    # both analytically and comparing with the SkyDisk model
    m_geom_2 = WcsGeom.create(
        binsz=0.1, width=(6, 6), skydir=(0, 90), coordsys="GAL", proj="AIT"
    )
    coords = m_geom_2.get_coord()
    lon = coords.lon * u.deg
    lat = coords.lat * u.deg

    semi_major = 5 * u.deg
    model_2 = SkyEllipse(0 * u.deg, 90 * u.deg, semi_major, 0.0, 0.0 * u.deg)
    vals_2 = model_2(lon, lat)
    mymap_2 = Map.from_geom(m_geom_2, data=vals_2.value)

    disk = SkyDisk(lon_0="0 deg", lat_0="90 deg", r_0="5 deg")
    vals_disk = disk(lon, lat)
    mymap_disk = Map.from_geom(m_geom_2, data=vals_disk.value)

    solid_angle = 2 * np.pi * (1 - np.cos(5 * u.deg))
    assert_allclose(np.max(vals_2).value * solid_angle, 1)

    assert_allclose(
        np.sum(mymap_2.quantity * u.sr ** -1 * m_geom_2.solid_angle()),
        np.sum(mymap_disk.quantity * u.sr ** -1 * m_geom_2.solid_angle()),
    )


def test_sky_ellipse_edge():
    pytest.importorskip("astropy", minversion="3.1.1")
    r_0 = 2 * u.deg
    model = SkyEllipse(
        lon_0="0 deg", lat_0="0 deg", semi_major=r_0, e=0.5, theta="0 deg"
    )
    value_center = model(0 * u.deg, 0 * u.deg)
    value_edge = model(0 * u.deg, r_0)
    assert_allclose((value_edge / value_center).to_value(""), 0.5)

    edge = model.edge.quantity
    value_edge_pwidth = model(0 * u.deg, r_0 + edge / 2)
    assert_allclose((value_edge_pwidth / value_center).to_value(""), 0.05)

    value_edge_nwidth = model(0 * u.deg, r_0 - edge / 2)
    assert_allclose((value_edge_nwidth / value_center).to_value(""), 0.95)


def test_sky_shell():
    width = 2 * u.deg
    rad = 2 * u.deg
    model = SkyShell(lon_0="1 deg", lat_0="45 deg", radius=rad, width=width)
    lon = [1, 2, 4] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == "deg-2"
    desired = [55.979449, 57.831651, 94.919895]
    assert_allclose(val.to_value("sr-1"), desired)
    radius = model.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, rad.value + width.value)


def test_sky_diffuse_constant():
    model = SkyDiffuseConstant(value="42 sr-1")
    lon = [1, 2] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == "sr-1"
    assert_allclose(val.value, 42)
    radius = model.evaluation_radius
    assert radius is None


@requires_data()
def test_sky_diffuse_map():
    filename = "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/RXJ1713_2016_250GeV.fits"
    model = SkyDiffuseMap.read(filename, normalize=False)
    lon = [258.5, 0] * u.deg
    lat = -39.8 * u.deg
    val = model(lon, lat)
    assert val.unit == "sr-1"
    desired = [3269.178107, 0]
    assert_allclose(val.value, desired)
    radius = model.evaluation_radius
    assert radius.unit == "deg"
    assert_allclose(radius.value, 0.64, rtol=1.0e-2)
    assert model.frame == "fk5"


@requires_data()
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

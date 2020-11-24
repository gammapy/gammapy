# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.maps import MapAxis, RegionGeom


@pytest.fixture()
def region():
    center = SkyCoord("0 deg", "0 deg", frame="galactic")
    return CircleSkyRegion(center=center, radius=1 * u.deg)


@pytest.fixture()
def energy_axis():
    return MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)


@pytest.fixture()
def test_axis():
    return MapAxis.from_nodes([1, 2], unit="", name="test")


def test_create(region):
    geom = RegionGeom.create(region)
    assert geom.frame == "galactic"
    assert geom.projection == "TAN"
    assert not geom.is_image
    assert not geom.is_allsky


def test_centers(region):
    geom = RegionGeom.create(region)
    assert_allclose(geom.center_skydir.l.deg, 0)
    assert_allclose(geom.center_skydir.b.deg, 0)
    assert_allclose(geom.center_pix, (0, 0))

    values = [_.value for _ in geom.center_coord]
    assert_allclose(values, (0, 0))


def test_width(region):
    geom = RegionGeom.create(region)
    assert_allclose(geom.width.value, [2.02, 2.02])


def test_create_axis(region, energy_axis, test_axis):
    geom = RegionGeom.create(region, axes=[energy_axis])

    assert geom.ndim == 3
    assert len(geom.axes) == 1
    assert geom.data_shape == (3, 1, 1)

    geom = RegionGeom.create(region, axes=[energy_axis, test_axis])
    assert geom.ndim == 4
    assert len(geom.axes) == 2
    assert geom.data_shape == (2, 3, 1, 1)


def test_get_coord(region, energy_axis, test_axis):
    geom = RegionGeom.create(region, axes=[energy_axis])
    coords = geom.get_coord()

    assert_allclose(coords.lon, 0)
    assert_allclose(coords.lat, 0)
    assert_allclose(
        coords["energy"].value.squeeze(), [1.467799, 3.162278, 6.812921], rtol=1e-5
    )

    geom = RegionGeom.create(region, axes=[energy_axis, test_axis])
    coords = geom.get_coord()
    assert coords["lon"].shape == (2, 3, 1, 1)
    assert coords["test"].shape == (2, 3, 1, 1)
    assert_allclose(
        coords["energy"].value[1].squeeze(), [1.467799, 3.162278, 6.812921], rtol=1e-5
    )
    assert_allclose(coords["test"].value[:, 1].squeeze(), [1, 2], rtol=1e-5)


def test_get_idx(region, energy_axis, test_axis):
    geom = RegionGeom.create(region, axes=[energy_axis])
    pix = geom.get_idx()

    assert_allclose(pix[0], 0)
    assert_allclose(pix[1], 0)
    assert_allclose(pix[2].squeeze(), [0, 1, 2])

    geom = RegionGeom.create(region, axes=[energy_axis, test_axis])
    pix = geom.get_idx()

    assert pix[0].shape == (2, 3, 1, 1)
    assert_allclose(pix[0], 0)
    assert_allclose(pix[1], 0)
    assert_allclose(pix[2][0].squeeze(), [0, 1, 2])


def test_coord_to_pix(region, energy_axis, test_axis):
    geom = RegionGeom.create(region, axes=[energy_axis])

    position = SkyCoord(0, 0, frame="galactic", unit="deg")
    coords = {"skycoord": position, "energy": 1 * u.TeV}
    coords_pix = geom.coord_to_pix(coords)

    assert_allclose(coords_pix[0], 0)
    assert_allclose(coords_pix[1], 0)
    assert_allclose(coords_pix[2], -0.5)

    geom = RegionGeom.create(region, axes=[energy_axis, test_axis])
    coords["test"] = 2
    coords_pix = geom.coord_to_pix(coords)

    assert_allclose(coords_pix[0], 0)
    assert_allclose(coords_pix[1], 0)
    assert_allclose(coords_pix[2], -0.5)
    assert_allclose(coords_pix[3], 1)


def test_pix_to_coord(region, energy_axis):
    geom = RegionGeom.create(region, axes=[energy_axis])

    pix = (0, 0, 0)
    coords = geom.pix_to_coord(pix)
    assert_allclose(coords[0].value, 0)
    assert_allclose(coords[1].value, 0)
    assert_allclose(coords[2].value, 1.467799, rtol=1e-5)

    pix = (1, 1, 1)
    coords = geom.pix_to_coord(pix)
    assert_allclose(coords[0].value, np.nan)
    assert_allclose(coords[1].value, np.nan)
    assert_allclose(coords[2].value, 3.162278, rtol=1e-5)

    pix = (1, 1, 3)
    coords = geom.pix_to_coord(pix)
    assert_allclose(coords[2].value, 14.677993, rtol=1e-5)


def test_pix_to_coord_2axes(region, energy_axis, test_axis):
    geom = RegionGeom.create(region, axes=[energy_axis, test_axis])

    pix = (0, 0, 0, 0)
    coords = geom.pix_to_coord(pix)
    assert_allclose(coords[0].value, 0)
    assert_allclose(coords[1].value, 0)
    assert_allclose(coords[2].value, 1.467799, rtol=1e-5)
    assert_allclose(coords[3].value, 1)

    pix = (0, 0, 0, 2)
    coords = geom.pix_to_coord(pix)
    assert_allclose(coords[3].value, 3)


def test_contains(region):
    geom = RegionGeom.create(region)
    position = SkyCoord([0, 0], [0, 1.1], frame="galactic", unit="deg")

    contains = geom.contains(coords={"skycoord": position})
    assert_allclose(contains, [1, 0])


def test_solid_angle(region):
    geom = RegionGeom.create(region)
    omega = geom.solid_angle()

    assert omega.unit == "sr"
    reference = 2 * np.pi * (1 - np.cos(region.radius))
    assert_allclose(omega.value, reference.value, rtol=1e-3)


def test_bin_volume(region):
    axis = MapAxis.from_edges([1, 3] * u.TeV, name="energy", interp="log")
    geom = RegionGeom.create(region, axes=[axis])
    volume = geom.bin_volume()

    assert volume.unit == "sr TeV"
    reference = 2 * 2 * np.pi * (1 - np.cos(region.radius))
    assert_allclose(volume.value, reference.value, rtol=1e-3)


def test_separation(region):
    geom = RegionGeom.create(region)

    position = SkyCoord([0, 0], [0, 1.1], frame="galactic", unit="deg")
    separation = geom.separation(position)

    assert_allclose(separation.deg, [0, 1.1])


def test_upsample(region):
    axis = MapAxis.from_edges([1, 10] * u.TeV, name="energy", interp="log")
    geom = RegionGeom.create(region, axes=[axis])
    geom_up = geom.upsample(factor=2, axis_name="energy")

    assert_allclose(geom_up.axes[0].edges.value, [1.0, 3.162278, 10.0], rtol=1e-5)


def test_downsample(region):
    axis = MapAxis.from_edges([1, 3.162278, 10] * u.TeV, name="energy", interp="log")
    geom = RegionGeom.create(region, axes=[axis])
    geom_down = geom.downsample(factor=2, axis_name="energy")

    assert_allclose(geom_down.axes[0].edges.value, [1.0, 10.0], rtol=1e-5)


def test_repr(region):
    axis = MapAxis.from_edges([1, 3.162278, 10] * u.TeV, name="energy", interp="log")
    geom = RegionGeom.create(region, axes=[axis])

    assert "RegionGeom" in repr(geom)
    assert "CircleSkyRegion" in repr(geom)


def test_eq(region):
    axis = MapAxis.from_edges([1, 10] * u.TeV, name="energy", interp="log")
    geom_1 = RegionGeom.create(region, axes=[axis])
    geom_2 = RegionGeom.create(region, axes=[axis])

    assert geom_1 == geom_2

    axis = MapAxis.from_edges([1, 100] * u.TeV, name="energy", interp="log")
    geom_3 = RegionGeom.create(region, axes=[axis])

    assert not geom_2 == geom_3


def test_to_cube_to_image(region):
    axis = MapAxis.from_edges([1, 10] * u.TeV, name="energy", interp="log")
    geom = RegionGeom.create(region)

    geom_cube = geom.to_cube([axis])
    assert geom_cube.ndim == 3

    geom = geom_cube.to_image()
    assert geom.ndim == 2


def test_squash(region):
    axis1 = MapAxis.from_edges([1, 10, 100] * u.TeV, name="energy", interp="log")
    axis2 = MapAxis.from_edges([1, 2, 3, 4] * u.deg, name="angle", interp="lin")

    geom = RegionGeom(region, axes=[axis1, axis2])

    geom_squashed = geom.squash("energy")

    assert len(geom_squashed.axes) == 2
    assert geom_squashed.axes[1] == axis2
    assert_allclose(geom_squashed.axes[0].edges.to_value("TeV"), (1, 100))

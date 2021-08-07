# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion, RectangleSkyRegion
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.utils.testing import mpl_plot_check, requires_dependency


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
    assert geom.is_image
    assert not geom.is_allsky


def test_binsz(region):
    geom = RegionGeom.create(region, binsz_wcs=0.05)
    wcs_geom = geom.to_wcs_geom()
    assert geom.binsz_wcs[0].deg == 0.05
    assert_allclose(wcs_geom.pixel_scales, geom.binsz_wcs)


def test_defined_wcs(region):
    wcs = WcsGeom.create(
        skydir=(0,0),
        frame="galactic",
        width="1.5deg",
        binsz="0.1deg").wcs
    geom = RegionGeom.create(region, wcs=wcs)
    assert geom.binsz_wcs[0].deg == 0.1


def test_to_binsz_wcs(region):
    binsz = 0.05*u.deg
    geom = RegionGeom.create(region, binsz_wcs=0.01)
    new_geom = geom.to_binsz_wcs(binsz)
    assert geom.binsz_wcs[0].deg == 0.01
    assert new_geom.binsz_wcs[0].deg == binsz.value


def test_centers(region):
    geom = RegionGeom.create(region)
    assert_allclose(geom.center_skydir.l.deg, 0, atol=1e-30)
    assert_allclose(geom.center_skydir.b.deg, 0, atol=1e-30)
    assert_allclose(geom.center_pix, (0, 0))

    values = [_.value for _ in geom.center_coord]
    assert_allclose(values, (0, 0), atol=1e-30)


def test_width(region):
    geom = RegionGeom.create(region, binsz_wcs=0.01)
    assert_allclose(geom.width.value, [2.02, 2.02])


def test_create_axis(region, energy_axis, test_axis):
    geom = RegionGeom.create(region, axes=[energy_axis])

    assert geom.ndim == 3
    assert len(geom.axes) == 1
    assert geom.data_shape == (3, 1, 1)
    assert geom.data_shape_axes == (3, 1, 1)

    geom = RegionGeom.create(region, axes=[energy_axis, test_axis])
    assert geom.ndim == 4
    assert len(geom.axes) == 2
    assert geom.data_shape == (2, 3, 1, 1)


def test_get_coord(region, energy_axis, test_axis):
    geom = RegionGeom.create(region, axes=[energy_axis])
    coords = geom.get_coord()

    assert_allclose(coords.lon, 0, atol=1e-30)
    assert_allclose(coords.lat, 0, atol=1e-30)
    assert_allclose(
        coords["energy"].value.squeeze(), [1.467799, 3.162278, 6.812921], rtol=1e-5
    )

    geom = RegionGeom.create(region, axes=[energy_axis, test_axis])
    coords = geom.get_coord(sparse=True)
    assert coords["lon"].shape == (1, 1)
    assert coords["test"].shape == (2, 1, 1, 1)
    assert coords["energy"].shape == (1, 3, 1, 1)

    assert_allclose(
        coords["energy"].value[0, :, 0, 0], [1.467799, 3.162278, 6.812921], rtol=1e-5
    )

    assert_allclose(coords["test"].value[:, 0, 0, 0].squeeze(), [1, 2], rtol=1e-5)


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
    assert_allclose(coords[0].value, 0, atol=1e-30)
    assert_allclose(coords[1].value, 0, atol=1e-30)
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
    assert_allclose(coords[0].value, 0, atol=1e-30)
    assert_allclose(coords[1].value, 0, atol=1e-30)
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

    assert_allclose(separation.deg, [0, 1.1], atol=1e-30)


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


def test_pad(region):
    axis1 = MapAxis.from_edges([1, 10] * u.TeV, name="energy", interp="log")
    axis2 = MapAxis.from_nodes([1, 2, 3, 4] * u.deg, name="angle", interp="lin")

    geom = RegionGeom(region, axes=[axis1, axis2])

    geom_pad = geom.pad(axis_name="energy", pad_width=1)
    assert_allclose(geom_pad.axes["energy"].nbin, 3)

    geom_pad = geom.pad(axis_name="angle", pad_width=1)
    assert_allclose(geom_pad.axes["angle"].nbin, 6)


def test_to_wcs_geom(region):
    geom = RegionGeom(region)
    wcs_geom = geom.to_wcs_geom()
    assert_allclose(wcs_geom.center_coord[1].value, 0, rtol=0.001, atol=0)
    assert_allclose(wcs_geom.width[0], 360 * u.deg, rtol=1, atol=0)
    assert wcs_geom.wcs.wcs.ctype[1] == 'GLAT-TAN'

    # test with an extra axis
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=10)
    geom_cube = geom.to_cube([axis])
    wcs_geom_cube = geom_cube.to_wcs_geom()
    assert wcs_geom_cube.to_image() == wcs_geom
    assert wcs_geom_cube.axes[0] == axis

    # test with minimum widths
    width_min = 3 * u.deg
    wcs_geom = geom.to_wcs_geom(width_min=width_min)
    assert_allclose(wcs_geom.center_coord[1].value, 0, rtol=0.001, atol=0)
    assert_allclose(wcs_geom.width, [[3], [3]] * u.deg, rtol=1, atol=0)

    width_min = [1, 3] * u.deg
    wcs_geom = geom.to_wcs_geom(width_min=width_min)
    assert_allclose(wcs_geom.center_coord[1].value, 0, rtol=0.001, atol=0)
    assert_allclose(wcs_geom.width, [[2], [3]] * u.deg, rtol=1, atol=0)


def test_get_wcs_coord_and_weights(region):
    # test on circular region
    geom = RegionGeom(region)
    region_coord, weights = geom.get_wcs_coord_and_weights()

    wcs_geom = geom.to_wcs_geom()
    solid_angles = wcs_geom.solid_angle().T[wcs_geom.coord_to_idx(region_coord)]
    area = (weights*solid_angles).sum()
    assert_allclose(area.value, geom.solid_angle().value, rtol=1e-3)

    assert region_coord.shape == weights.shape

    # test on rectangular region (assymetric)
    center = SkyCoord("0 deg", "0 deg", frame="galactic")
    region = RectangleSkyRegion(center=center, width=1 * u.deg, height=2 * u.deg, angle=15*u.deg)
    geom = RegionGeom(region)

    wcs_geom = geom.to_wcs_geom()
    region_coord, weights = geom.get_wcs_coord_and_weights()
    solid_angles = wcs_geom.solid_angle().T[wcs_geom.coord_to_idx(region_coord)]
    area = (weights * solid_angles).sum()
    assert_allclose(area.value, geom.solid_angle().value, rtol=1e-3)
    assert region_coord.shape == weights.shape


@requires_dependency("matplotlib")
def test_region_nd_map_plot(region):
    import matplotlib.pyplot as plt
    geom = RegionGeom(region)

    ax = plt.subplot(projection=geom.wcs)
    with mpl_plot_check():
        geom.plot_region(ax=ax)

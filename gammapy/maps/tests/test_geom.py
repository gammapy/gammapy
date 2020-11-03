# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.maps import MapAxis, MapCoord

mapaxis_geoms = [
    (np.array([0.25, 0.75, 1.0, 2.0]), "lin"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "log"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "sqrt"),
]

mapaxis_geoms_node_type = [
    ([0.25, 0.75, 1.0, 2.0], "lin", "edges"),
    ([0.25, 0.75, 1.0, 2.0], "log", "edges"),
    ([0.25, 0.75, 1.0, 2.0], "sqrt", "edges"),
    ([0.25, 0.75, 1.0, 2.0], "lin", "center"),
    ([0.25, 0.75, 1.0, 2.0], "log", "center"),
    ([0.25, 0.75, 1.0, 2.0], "sqrt", "center"),
]


@pytest.mark.parametrize(("edges", "interp"), mapaxis_geoms)
def test_mapaxis_init_from_edges(edges, interp):
    axis = MapAxis(edges, interp=interp)
    assert_allclose(axis.edges, edges)
    assert_allclose(axis.nbin, len(edges) - 1)
    with pytest.raises(ValueError):
        MapAxis.from_edges([1])
        MapAxis.from_edges([0, 1, 1, 2])
        MapAxis.from_edges([0, 1, 3, 2])


@pytest.mark.parametrize(("nodes", "interp"), mapaxis_geoms)
def test_mapaxis_from_nodes(nodes, interp):
    axis = MapAxis.from_nodes(nodes, interp=interp)
    assert_allclose(axis.center, nodes)
    assert_allclose(axis.nbin, len(nodes))
    with pytest.raises(ValueError):
        MapAxis.from_nodes([])
        MapAxis.from_nodes([0, 1, 1, 2])
        MapAxis.from_nodes([0, 1, 3, 2])


@pytest.mark.parametrize(("nodes", "interp"), mapaxis_geoms)
def test_mapaxis_from_bounds(nodes, interp):
    axis = MapAxis.from_bounds(nodes[0], nodes[-1], 3, interp=interp)
    assert_allclose(axis.edges[0], nodes[0])
    assert_allclose(axis.edges[-1], nodes[-1])
    assert_allclose(axis.nbin, 3)
    with pytest.raises(ValueError):
        MapAxis.from_bounds(1, 1, 1)


@pytest.mark.parametrize(("nodes", "interp", "node_type"), mapaxis_geoms_node_type)
def test_mapaxis_pix_to_coord(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    assert_allclose(axis.center, axis.pix_to_coord(np.arange(axis.nbin, dtype=float)))
    assert_allclose(
        np.arange(axis.nbin + 1, dtype=float) - 0.5, axis.coord_to_pix(axis.edges)
    )


@pytest.mark.parametrize(("nodes", "interp", "node_type"), mapaxis_geoms_node_type)
def test_mapaxis_coord_to_idx(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    assert_allclose(np.arange(axis.nbin, dtype=int), axis.coord_to_idx(axis.center))


@pytest.mark.parametrize(("nodes", "interp", "node_type"), mapaxis_geoms_node_type)
def test_mapaxis_slice(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    saxis = axis.slice(slice(1, 3))
    assert_allclose(saxis.nbin, 2)
    assert_allclose(saxis.center, axis.center[slice(1, 3)])

    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    saxis = axis.slice(slice(1, None))
    assert_allclose(saxis.nbin, axis.nbin - 1)
    assert_allclose(saxis.center, axis.center[slice(1, None)])

    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    saxis = axis.slice(slice(None, 2))
    assert_allclose(saxis.nbin, 2)
    assert_allclose(saxis.center, axis.center[slice(None, 2)])

    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    saxis = axis.slice(slice(None, -1))
    assert_allclose(saxis.nbin, axis.nbin - 1)
    assert_allclose(saxis.center, axis.center[slice(None, -1)])


def test_mapcoords_create():
    # From existing MapCoord
    coords_cel = MapCoord.create((0.0, 1.0), frame="icrs")
    coords_gal = MapCoord.create(coords_cel, frame="galactic")
    assert_allclose(coords_gal.lon, coords_cel.skycoord.galactic.l.deg)
    assert_allclose(coords_gal.lat, coords_cel.skycoord.galactic.b.deg)

    # 2D Tuple of scalars
    coords = MapCoord.create((0.0, 1.0))
    assert_allclose(coords.lon, 0.0)
    assert_allclose(coords.lat, 1.0)
    assert_allclose(coords[0], 0.0)
    assert_allclose(coords[1], 1.0)
    assert coords.frame is None
    assert coords.ndim == 2

    # 3D Tuple of scalars
    coords = MapCoord.create((0.0, 1.0, 2.0))
    assert_allclose(coords[0], 0.0)
    assert_allclose(coords[1], 1.0)
    assert_allclose(coords[2], 2.0)
    assert coords.frame is None
    assert coords.ndim == 3

    # 2D Tuple w/ NaN coordinates
    coords = MapCoord.create((np.nan, np.nan))

    # 2D Tuple w/ NaN coordinates
    lon, lat = np.array([np.nan, 1.0]), np.array([np.nan, 3.0])
    coords = MapCoord.create((lon, lat))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)

    # 2D Tuple w/ SkyCoord
    lon, lat = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    energy = np.array([100.0, 1000.0])
    skycoord_cel = SkyCoord(lon, lat, unit="deg", frame="icrs")
    skycoord_gal = SkyCoord(lon, lat, unit="deg", frame="galactic")
    coords = MapCoord.create((skycoord_cel,))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.frame == "icrs"
    assert coords.ndim == 2

    coords = MapCoord.create((skycoord_gal,))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.frame == "galactic"
    assert coords.ndim == 2

    # SkyCoord
    coords = MapCoord.create(skycoord_cel)
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.frame == "icrs"
    assert coords.ndim == 2
    coords = MapCoord.create(skycoord_gal)
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.frame == "galactic"
    assert coords.ndim == 2

    # 2D dict w/ vectors
    coords = MapCoord.create(dict(lon=lon, lat=lat))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.ndim == 2

    # 3D dict w/ vectors
    coords = MapCoord.create(dict(lon=lon, lat=lat, energy=energy))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert_allclose(coords["energy"], energy)
    assert coords.ndim == 3

    # 3D dict w/ SkyCoord
    coords = MapCoord.create(dict(skycoord=skycoord_cel, energy=energy))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert_allclose(coords["energy"], energy)
    assert coords.frame == "icrs"
    assert coords.ndim == 3

    # 3D dict  w/ vectors
    coords = MapCoord.create({"energy": energy, "lat": lat, "lon": lon})
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert_allclose(coords["energy"], energy)
    assert_allclose(coords[0], energy)
    assert_allclose(coords[1], lat)
    assert_allclose(coords[2], lon)
    assert coords.ndim == 3

    # Quantities
    coords = MapCoord.create(dict(energy=energy * u.TeV, lat=lat, lon=lon))
    assert coords["energy"].unit == "TeV"


def test_mapcoords_to_frame():
    lon, lat = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    energy = np.array([100.0, 1000.0])
    skycoord_cel = SkyCoord(lon, lat, unit="deg", frame="icrs")
    skycoord_gal = SkyCoord(lon, lat, unit="deg", frame="galactic")

    coords = MapCoord.create(dict(lon=lon, lat=lat, energy=energy), frame="icrs")
    assert coords.frame == "icrs"
    assert_allclose(coords.skycoord.transform_to("icrs").ra.deg, skycoord_cel.ra.deg)
    assert_allclose(coords.skycoord.transform_to("icrs").dec.deg, skycoord_cel.dec.deg)
    coords = coords.to_frame("galactic")
    assert coords.frame == "galactic"
    assert_allclose(
        coords.skycoord.transform_to("galactic").l.deg, skycoord_cel.galactic.l.deg
    )
    assert_allclose(
        coords.skycoord.transform_to("galactic").b.deg, skycoord_cel.galactic.b.deg
    )

    coords = MapCoord.create(dict(lon=lon, lat=lat, energy=energy), frame="galactic")
    assert coords.frame == "galactic"
    assert_allclose(coords.skycoord.transform_to("galactic").l.deg, skycoord_gal.l.deg)
    assert_allclose(coords.skycoord.transform_to("galactic").b.deg, skycoord_gal.b.deg)
    coords = coords.to_frame("icrs")
    assert coords.frame == "icrs"
    assert_allclose(
        coords.skycoord.transform_to("icrs").ra.deg, skycoord_gal.icrs.ra.deg
    )
    assert_allclose(
        coords.skycoord.transform_to("icrs").dec.deg, skycoord_gal.icrs.dec.deg
    )


def test_mapaxis_repr():
    axis = MapAxis([1, 2, 3], name="test")
    assert "MapAxis" in repr(axis)


def test_mapcoord_repr():
    coord = MapCoord({"lon": 0, "lat": 0, "energy": 5})
    assert "MapCoord" in repr(coord)


nodes_array = np.array([0.25, 0.75, 1.0, 2.0])
mapaxis_geoms_node_type_unit = [
    (nodes_array, "lin", "edges", "s", "TEST", True),
    (nodes_array, "log", "edges", "s", "test", False),
    (nodes_array, "lin", "edges", "TeV", "TEST", False),
    (nodes_array, "sqrt", "edges", "s", "test", False),
    (nodes_array, "lin", "center", "s", "test", False),
    (nodes_array + 1e-9, "lin", "edges", "s", "test", True),
    (nodes_array + 1e-3, "lin", "edges", "s", "test", False),
    (nodes_array / 3600.0, "lin", "edges", "hr", "TEST", True),
]


@pytest.mark.parametrize(
    ("nodes", "interp", "node_type", "unit", "name", "result"),
    mapaxis_geoms_node_type_unit,
)
def test_mapaxis_equal(nodes, interp, node_type, unit, name, result):
    axis1 = MapAxis(nodes_array, name="test", unit="s", interp="lin", node_type="edges")

    axis2 = MapAxis(nodes, name=name, unit=unit, interp=interp, node_type=node_type)

    assert (axis1 == axis2) is result
    assert (axis1 != axis2) is not result


def test_squash():
    axis = MapAxis(
        nodes=[0, 1, 2, 3], unit="TeV", name="energy", node_type="edges", interp="lin"
    )
    ax_sq = axis.squash()

    assert_allclose(ax_sq.nbin, 1)
    assert_allclose(axis.edges[0], ax_sq.edges[0])
    assert_allclose(axis.edges[-1], ax_sq.edges[1])
    assert_allclose(ax_sq.center, 1.5 * u.TeV)


def test_upsample():
    axis = MapAxis(
        nodes=[0, 1, 2, 3], unit="TeV", name="energy", node_type="edges", interp="lin"
    )
    axis_up = axis.upsample(10)

    assert_allclose(axis_up.nbin, 10 * axis.nbin)
    assert_allclose(axis_up.edges[0], axis.edges[0])
    assert_allclose(axis_up.edges[-1], axis.edges[-1])
    assert axis_up.node_type == axis.node_type


def test_downsample():
    axis = MapAxis(
        nodes=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        unit="TeV",
        name="energy",
        node_type="edges",
        interp="lin",
    )
    axis_down = axis.downsample(2)

    assert_allclose(axis_down.nbin, 0.5 * axis.nbin)
    assert_allclose(axis_down.edges[0], axis.edges[0])
    assert_allclose(axis_down.edges[-1], axis.edges[-1])
    assert axis_down.node_type == axis.node_type


def test_upsample_non_regular():
    axis = MapAxis.from_edges([0, 1, 3, 7], name="test", interp="lin")
    axis_up = axis.upsample(2)

    assert_allclose(axis_up.nbin, 2 * axis.nbin)
    assert_allclose(axis_up.edges[0], axis.edges[0])
    assert_allclose(axis_up.edges[-1], axis.edges[-1])
    assert axis_up.node_type == axis.node_type


def test_upsample_non_regular_nodes():
    axis = MapAxis.from_nodes([0, 1, 3, 7], name="test", interp="lin")
    axis_up = axis.upsample(2)

    assert_allclose(axis_up.nbin, 2 * axis.nbin - 1)
    assert_allclose(axis_up.center[0], axis.center[0])
    assert_allclose(axis_up.center[-1], axis.center[-1])
    assert axis_up.node_type == axis.node_type


def test_downsample_non_regular():
    axis = MapAxis.from_edges([0, 1, 3, 7, 13], name="test", interp="lin")
    axis_down = axis.downsample(2)

    assert_allclose(axis_down.nbin, 0.5 * axis.nbin)
    assert_allclose(axis_down.edges[0], axis.edges[0])
    assert_allclose(axis_down.edges[-1], axis.edges[-1])
    assert axis_down.node_type == axis.node_type


def test_downsample_non_regular_nodes():
    axis = MapAxis.from_edges([0, 1, 3, 7, 9], name="test", interp="lin")
    axis_down = axis.downsample(2)

    assert_allclose(axis_down.nbin, 0.5 * axis.nbin)
    assert_allclose(axis_down.edges[0], axis.edges[0])
    assert_allclose(axis_down.edges[-1], axis.edges[-1])
    assert axis_down.node_type == axis.node_type


@pytest.mark.parametrize("factor", [1, 3, 5, 7, 11])
def test_up_downsample_consistency(factor):
    axis = MapAxis.from_edges([0, 1, 3, 7, 13], name="test", interp="lin")
    axis_new = axis.upsample(factor).downsample(factor)
    assert_allclose(axis.edges, axis_new.edges)


@pytest.fixture(scope="session")
def energy_axis_ref():
    edges = np.arange(1, 11) * u.TeV
    return MapAxis.from_edges(edges, name="energy")


def test_group_table_basic(energy_axis_ref):
    energy_edges = [1, 2, 10] * u.TeV

    groups = energy_axis_ref.group_table(energy_edges)

    assert_allclose(groups["group_idx"], [0, 1])
    assert_allclose(groups["idx_min"], [0, 1])
    assert_allclose(groups["idx_max"], [0, 8])
    assert_allclose(groups["energy_min"], [1, 2])
    assert_allclose(groups["energy_max"], [2, 10])

    bin_type = [_.strip() for _ in groups["bin_type"]]
    assert_equal(bin_type, ["normal", "normal"])


@pytest.mark.parametrize(
    "energy_edges",
    [[1.8, 4.8, 7.2] * u.TeV, [2, 5, 7] * u.TeV, [2000, 5000, 7000] * u.GeV],
)
def test_group_tablenergy_edges(energy_axis_ref, energy_edges):
    groups = energy_axis_ref.group_table(energy_edges)

    assert_allclose(groups["group_idx"], [0, 1, 2, 3])
    assert_allclose(groups["idx_min"], [0, 1, 4, 6])
    assert_allclose(groups["idx_max"], [0, 3, 5, 8])
    assert_allclose(groups["energy_min"].quantity.to_value("TeV"), [1, 2, 5, 7])
    assert_allclose(groups["energy_max"].quantity.to_value("TeV"), [2, 5, 7, 10])

    bin_type = [_.strip() for _ in groups["bin_type"]]
    assert_equal(bin_type, ["underflow", "normal", "normal", "overflow"])


def test_group_table_below_range(energy_axis_ref):
    energy_edges = [0.7, 0.8, 1, 4] * u.TeV
    groups = energy_axis_ref.group_table(energy_edges)

    assert_allclose(groups["group_idx"], [0, 1])
    assert_allclose(groups["idx_min"], [0, 3])
    assert_allclose(groups["idx_max"], [2, 8])
    assert_allclose(groups["energy_min"], [1, 4])
    assert_allclose(groups["energy_max"], [4, 10])

    bin_type = [_.strip() for _ in groups["bin_type"]]
    assert_equal(bin_type, ["normal", "overflow"])


def test_group_table_above_range(energy_axis_ref):
    energy_edges = [5, 7, 11, 13] * u.TeV
    groups = energy_axis_ref.group_table(energy_edges)

    assert_allclose(groups["group_idx"], [0, 1, 2])
    assert_allclose(groups["idx_min"], [0, 4, 6])
    assert_allclose(groups["idx_max"], [3, 5, 8])
    assert_allclose(groups["energy_min"], [1, 5, 7])
    assert_allclose(groups["energy_max"], [5, 7, 10])

    bin_type = [_.strip() for _ in groups["bin_type"]]
    assert_equal(bin_type, ["underflow", "normal", "normal"])


def test_group_table_outside_range(energy_axis_ref):
    energy_edges = [20, 30, 40] * u.TeV

    with pytest.raises(ValueError):
        energy_axis_ref.group_table(energy_edges)


def test_map_axis_single_bin():
    with pytest.raises(ValueError):
        _ = MapAxis.from_nodes([1])


def test_map_axis_aligned():
    ax1 = MapAxis([1, 2, 3], interp="lin", node_type="edges")
    ax2 = MapAxis([1.5, 2.5], interp="log", node_type="center")
    assert not ax1.is_aligned(ax2)

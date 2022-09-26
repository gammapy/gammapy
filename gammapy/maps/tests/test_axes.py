# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from gammapy.data import GTI
from gammapy.maps import LabelMapAxis, MapAxes, MapAxis, RegionNDMap, TimeMapAxis
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import assert_time_allclose, mpl_plot_check, requires_data
from gammapy.utils.time import time_ref_to_dict

MAP_AXIS_INTERP = [
    (np.array([0.25, 0.75, 1.0, 2.0]), "lin"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "log"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "sqrt"),
]

MAP_AXIS_NODE_TYPES = [
    ([0.25, 0.75, 1.0, 2.0], "lin", "edges"),
    ([0.25, 0.75, 1.0, 2.0], "log", "edges"),
    ([0.25, 0.75, 1.0, 2.0], "sqrt", "edges"),
    ([0.25, 0.75, 1.0, 2.0], "lin", "center"),
    ([0.25, 0.75, 1.0, 2.0], "log", "center"),
    ([0.25, 0.75, 1.0, 2.0], "sqrt", "center"),
]


nodes_array = np.array([0.25, 0.75, 1.0, 2.0])

MAP_AXIS_NODE_TYPE_UNIT = [
    (nodes_array, "lin", "edges", "s", "TEST", True),
    (nodes_array, "log", "edges", "s", "test", False),
    (nodes_array, "lin", "edges", "TeV", "TEST", False),
    (nodes_array, "sqrt", "edges", "s", "test", False),
    (nodes_array, "lin", "center", "s", "test", False),
    (nodes_array + 1e-9, "lin", "edges", "s", "test", True),
    (nodes_array + 1e-3, "lin", "edges", "s", "test", False),
    (nodes_array / 3600.0, "lin", "edges", "hr", "TEST", True),
]


@pytest.fixture
def time_intervals():
    t0 = Time("2020-03-19")
    t_min = np.linspace(0, 10, 20) * u.d
    t_max = t_min + 1 * u.h
    return {"t_min": t_min, "t_max": t_max, "t_ref": t0}


@pytest.fixture
def time_interval():
    t0 = Time("2020-03-19")
    t_min = 1 * u.d
    t_max = 11 * u.d
    return {"t_min": t_min, "t_max": t_max, "t_ref": t0}


@pytest.fixture(scope="session")
def energy_axis_ref():
    edges = np.arange(1, 11) * u.TeV
    return MapAxis.from_edges(edges, name="energy")


def test_mapaxis_repr():
    axis = MapAxis([1, 2, 3], name="test")
    assert "MapAxis" in repr(axis)


def test_mapaxis_invalid_name():
    with pytest.raises(TypeError):
        MapAxis([1, 2, 3], name=1)


@pytest.mark.parametrize(
    ("nodes", "interp", "node_type", "unit", "name", "result"),
    MAP_AXIS_NODE_TYPE_UNIT,
)
def test_mapaxis_equal(nodes, interp, node_type, unit, name, result):
    axis1 = MapAxis(
        nodes=[0.25, 0.75, 1.0, 2.0],
        name="test",
        unit="s",
        interp="lin",
        node_type="edges",
    )

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


def test_one_bin_nodes():
    axis = MapAxis.from_nodes([1], name="test", unit="deg")

    assert_allclose(axis.center, 1 * u.deg)
    assert_allclose(axis.coord_to_pix(1 * u.deg), 0)
    assert_allclose(axis.coord_to_pix(2 * u.deg), 0)
    assert_allclose(axis.pix_to_coord(0), 1 * u.deg)


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


def test_map_axis_aligned():
    ax1 = MapAxis([1, 2, 3], interp="lin", node_type="edges")
    ax2 = MapAxis([1.5, 2.5], interp="log", node_type="center")
    assert not ax1.is_aligned(ax2)


def test_map_axis_pad():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)

    padded = axis.pad(pad_width=(0, 1))
    assert_allclose(padded.edges, [1, 10, 100] * u.TeV)

    padded = axis.pad(pad_width=(1, 0))
    assert_allclose(padded.edges, [0.1, 1, 10] * u.TeV)

    padded = axis.pad(pad_width=1)
    assert_allclose(padded.edges, [0.1, 1, 10, 100] * u.TeV)


def test_map_axes_pad():
    axis_1 = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    axis_2 = MapAxis.from_bounds(0, 1, nbin=2, unit="deg", name="rad")

    axes = MapAxes([axis_1, axis_2])

    axes = axes.pad(axis_name="energy", pad_width=1)

    assert_allclose(axes["energy"].edges, [0.1, 1, 10, 100] * u.TeV)


def test_rename():
    axis_1 = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    axis = axis_1.rename("energy_true")
    assert axis_1.name == "energy"
    assert axis.name == "energy_true"

    axis_2 = MapAxis.from_bounds(0, 1, nbin=2, unit="deg", name="rad")

    axes = MapAxes([axis_1, axis_2])
    axes = axes.rename_axes(["energy", "rad"], ["energy_true", "angle"])
    assert axes.names == ["energy_true", "angle"]


@pytest.mark.parametrize(("edges", "interp"), MAP_AXIS_INTERP)
def test_mapaxis_init_from_edges(edges, interp):
    axis = MapAxis(edges, interp=interp)
    assert_allclose(axis.edges, edges)
    assert_allclose(axis.nbin, len(edges) - 1)
    with pytest.raises(ValueError):
        MapAxis.from_edges([1])
        MapAxis.from_edges([0, 1, 1, 2])
        MapAxis.from_edges([0, 1, 3, 2])


@pytest.mark.parametrize(("nodes", "interp"), MAP_AXIS_INTERP)
def test_mapaxis_from_nodes(nodes, interp):
    axis = MapAxis.from_nodes(nodes, interp=interp)
    assert_allclose(axis.center, nodes)
    assert_allclose(axis.nbin, len(nodes))
    with pytest.raises(ValueError):
        MapAxis.from_nodes([])
        MapAxis.from_nodes([0, 1, 1, 2])
        MapAxis.from_nodes([0, 1, 3, 2])


@pytest.mark.parametrize(("nodes", "interp"), MAP_AXIS_INTERP)
def test_mapaxis_from_bounds(nodes, interp):
    axis = MapAxis.from_bounds(nodes[0], nodes[-1], 3, interp=interp)
    assert_allclose(axis.edges[0], nodes[0])
    assert_allclose(axis.edges[-1], nodes[-1])
    assert_allclose(axis.nbin, 3)
    with pytest.raises(ValueError):
        MapAxis.from_bounds(1, 1, 1)


def test_map_axis_from_energy_units():
    with pytest.raises(ValueError):
        _ = MapAxis.from_energy_bounds(0.1, 10, 2, unit="deg")

    with pytest.raises(ValueError):
        _ = MapAxis.from_energy_edges([0.1, 1, 10] * u.deg)


@pytest.mark.parametrize(("nodes", "interp", "node_type"), MAP_AXIS_NODE_TYPES)
def test_mapaxis_pix_to_coord(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    assert_allclose(axis.center, axis.pix_to_coord(np.arange(axis.nbin, dtype=float)))
    assert_allclose(
        np.arange(axis.nbin + 1, dtype=float) - 0.5, axis.coord_to_pix(axis.edges)
    )


@pytest.mark.parametrize(("nodes", "interp", "node_type"), MAP_AXIS_NODE_TYPES)
def test_mapaxis_coord_to_idx(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    assert_allclose(np.arange(axis.nbin, dtype=int), axis.coord_to_idx(axis.center))


@pytest.mark.parametrize(("nodes", "interp", "node_type"), MAP_AXIS_NODE_TYPES)
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


def test_map_axis_plot_helpers():
    axis = MapAxis.from_nodes([0, 1, 2], unit="deg", name="offset")
    labels = axis.as_plot_labels

    assert labels[0] == "0.00e+00 deg"

    assert_allclose(axis.center, axis.as_plot_center)
    assert_allclose(axis.edges, axis.as_plot_edges)


def test_time_axis(time_intervals):
    axis = TimeMapAxis(
        time_intervals["t_min"], time_intervals["t_max"], time_intervals["t_ref"]
    )

    axis_copy = axis.copy()

    assert axis.nbin == 20
    assert axis.name == "time"
    assert axis.node_type == "intervals"

    assert_allclose(axis.time_delta.to_value("min"), 60)
    assert_allclose(axis.time_mid[0].mjd, 58927.020833333336)

    assert "time" in axis.__str__()
    assert "20" in axis.__str__()

    with pytest.raises(ValueError):
        axis.assert_name("bad")

    assert axis_copy == axis

    assert not axis.is_contiguous

    ax_cont = axis.to_contiguous()
    assert_allclose(ax_cont.nbin, 39)


def test_single_interval_time_axis(time_interval):
    axis = TimeMapAxis(
        edges_min=time_interval["t_min"],
        edges_max=time_interval["t_max"],
        reference_time=time_interval["t_ref"],
    )

    coord = Time(58933, format="mjd") + u.Quantity([1.5, 3.5, 10], unit="d")
    pix = axis.coord_to_pix(coord)

    assert axis.nbin == 1
    assert_allclose(axis.time_delta.to_value("d"), 10)
    assert_allclose(axis.time_mid[0].mjd, 58933)

    pix_min = axis.coord_to_pix(time_interval["t_min"] + 0.001 * u.s)
    assert_allclose(pix_min, -0.5)

    pix_max = axis.coord_to_pix(time_interval["t_max"] - 0.001 * u.s)
    assert_allclose(pix_max, 0.5)

    assert_allclose(pix, [0.15, 0.35, np.nan])


def test_slice_squash_time_axis(time_intervals):
    axis = TimeMapAxis(
        time_intervals["t_min"], time_intervals["t_max"], time_intervals["t_ref"]
    )
    axis_squash = axis.squash()
    axis_slice = axis.slice(slice(1, 5))

    assert axis_squash.nbin == 1
    assert_allclose(axis_squash.time_min[0].mjd, 58927)
    assert_allclose(axis_squash.time_delta.to_value("d"), 10.04166666)
    assert axis_slice.nbin == 4
    assert_allclose(axis_slice.time_delta.to_value("d")[0], 0.04166666666)
    assert axis_squash != axis_slice


def test_from_time_edges_time_axis():
    t0 = Time("2020-03-19")
    t_min = t0 + np.linspace(0, 10, 20) * u.d
    t_max = t_min + 1 * u.h

    axis = TimeMapAxis.from_time_edges(t_min, t_max)
    axis_h = TimeMapAxis.from_time_edges(t_min, t_max, unit="h")

    assert axis.nbin == 20
    assert axis.name == "time"
    assert_time_allclose(axis.reference_time, t0)
    assert_allclose(axis.time_delta.to_value("min"), 60)
    assert_allclose(axis.time_mid[0].mjd, 58927.020833333336)
    assert_allclose(axis_h.time_delta.to_value("h"), 1)
    assert_allclose(axis_h.time_mid[0].mjd, 58927.020833333336)
    assert axis == axis_h


def test_incorrect_time_axis():
    tmin = np.linspace(0, 10, 20) * u.h
    tmax = np.linspace(1, 11, 20) * u.h

    # incorrect reference time
    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax, reference_time=51000 * u.d, name="time")

    # overlapping time intervals
    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax, reference_time=Time.now(), name="time")


def test_bad_length_sort_time_axis(time_intervals):
    tref = time_intervals["t_ref"]
    tmin = time_intervals["t_min"]
    tmax_reverse = time_intervals["t_max"][::-1]
    tmax_short = time_intervals["t_max"][:-1]

    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax_reverse, tref, name="time")

    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax_short, tref, name="time")


def test_coord_to_idx_time_axis(time_intervals):
    tmin = time_intervals["t_min"]
    tmax = time_intervals["t_max"]
    tref = time_intervals["t_ref"]
    axis = TimeMapAxis(tmin, tmax, tref, name="time")

    time = Time(58927.020833333336, format="mjd")

    times = axis.time_mid
    times[::2] += 1 * u.h
    times = times.insert(0, tref - [1, 2] * u.yr)

    idx = axis.coord_to_idx(time)
    indices = axis.coord_to_idx(times)

    pix = axis.coord_to_pix(time)
    pixels = axis.coord_to_pix(times)

    assert idx == 0
    assert_allclose(indices[1::2], [-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    assert_allclose(indices[::2], -1)
    assert_allclose(pix, 0, atol=1e-10)
    assert_allclose(pixels[1::2], [np.nan, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19])


def test_slice_time_axis(time_intervals):
    axis = TimeMapAxis(
        time_intervals["t_min"], time_intervals["t_max"], time_intervals["t_ref"]
    )

    new_axis = axis.slice([2, 6, 9])
    squashed = axis.squash()

    assert new_axis.nbin == 3
    assert_allclose(squashed.time_max[0].mjd, 58937.041667)
    assert squashed.nbin == 1
    assert_allclose(squashed.time_max[0].mjd, 58937.041667)


def test_time_map_axis_from_time_bounds():
    t_min = Time("2006-02-12", scale="utc")
    t_max = t_min + 12 * u.h

    axis = TimeMapAxis.from_time_bounds(time_min=t_min, time_max=t_max, nbin=3)
    assert_allclose(axis.center, [0.083333, 0.25, 0.416667] * u.d, rtol=1e-5)


def test_from_table_time_axis():
    t0 = Time("2006-02-12", scale="utc")
    t_min = np.linspace(0, 10, 10) * u.d
    t_max = t_min + 12 * u.h

    table = Table()
    table["TIME_MIN"] = t_min
    table["TIME_MAX"] = t_max
    table.meta.update(time_ref_to_dict(t0))
    table.meta["AXCOLS1"] = "TIME_MIN,TIME_MAX"

    axis = TimeMapAxis.from_table(table, format="gadf")

    assert axis.nbin == 10
    assert_allclose(axis.time_mid[0].mjd, 53778.25)


@requires_data()
def test_from_gti_time_axis():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    filename = make_path(filename)
    gti = GTI.read(filename)

    axis = TimeMapAxis.from_gti(gti)
    expected = Time(53090.123451203704, format="mjd", scale="tt")
    assert_time_allclose(axis.time_min[0], expected)
    assert axis.nbin == 1


def test_map_with_time_axis(time_intervals):
    time_axis = TimeMapAxis(
        time_intervals["t_min"], time_intervals["t_max"], time_intervals["t_ref"]
    )
    energy_axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit="TeV")
    region_map = RegionNDMap.create(
        region="fk5; circle(0,0,0.1)", axes=[energy_axis, time_axis]
    )

    assert region_map.geom.data_shape == (20, 2, 1, 1)


def test_time_axis_plot_helpers():
    time_ref = Time("1999-01-01T00:00:00.123456789")

    time_axis = TimeMapAxis(
        edges_min=[0, 1, 3] * u.d,
        edges_max=[0.8, 1.9, 5.4] * u.d,
        reference_time=time_ref,
    )

    labels = time_axis.as_plot_labels
    assert labels[0] == "1999-01-01 00:00:00.123 - 1999-01-01 19:12:00.123"

    center = time_axis.as_plot_center
    assert center[0].year == 1999

    edges = time_axis.to_contiguous().as_plot_edges
    assert edges[0].year == 1999


def test_axes_basics():
    energy_axis = MapAxis.from_energy_edges([1, 3] * u.TeV)

    time_ref = Time("1999-01-01T00:00:00.123456789")

    time_axis = TimeMapAxis(
        edges_min=[0, 1, 3] * u.d,
        edges_max=[0.8, 1.9, 5.4] * u.d,
        reference_time=time_ref,
    )

    axes = MapAxes([energy_axis, time_axis])

    assert axes.shape == (1, 3)
    assert axes.is_unidimensional
    assert not axes.is_flat

    assert axes.primary_axis.name == "time"

    new_axes = axes.copy()
    assert new_axes[0] == new_axes[0]
    assert new_axes[1] == new_axes[1]
    assert new_axes == axes

    energy_axis = MapAxis.from_energy_edges([1, 4] * u.TeV)
    new_axes = MapAxes([energy_axis, time_axis.copy()])
    assert new_axes != axes


def test_axes_getitem():
    axis1 = MapAxis.from_bounds(1, 4, 3, name="a1")
    axis2 = axis1.copy(name="a2")
    axis3 = axis1.copy(name="a3")
    axes = MapAxes([axis1, axis2, axis3])

    assert isinstance(axes[0], MapAxis)
    assert axes[-1].name == "a3"
    assert isinstance(axes[1:], MapAxes)
    assert len(axes[1:]) == 2
    assert isinstance(axes[0:1], MapAxes)
    assert len(axes[0:1]) == 1
    assert isinstance(axes[["a3", "a1"]], MapAxes)
    assert axes[["a3", "a1"]][0].name == "a3"


def test_label_map_axis_basics():
    axis = LabelMapAxis(labels=["label-1", "label-2"], name="label-axis")

    axis_str = str(axis)
    assert "node type" in axis_str
    assert "labels" in axis_str
    assert "label-2" in axis_str

    with pytest.raises(ValueError):
        axis.assert_name("time")

    assert axis.nbin == 2
    assert axis.node_type == "label"

    assert_allclose(axis.bin_width, 1)

    assert axis.name == "label-axis"

    with pytest.raises(ValueError):
        axis.edges

    axis_copy = axis.copy()
    assert axis_copy.name == "label-axis"


def test_label_map_axis_coord_to_idx():
    axis = LabelMapAxis(labels=["label-1", "label-2", "label-3"], name="label-axis")

    labels = "label-1"
    idx = axis.coord_to_idx(coord=labels)
    assert_allclose(idx, 0)

    labels = ["label-1", "label-3"]
    idx = axis.coord_to_idx(coord=labels)
    assert_allclose(idx, [0, 2])

    labels = [["label-1"], ["label-2"]]
    idx = axis.coord_to_idx(coord=labels)
    assert_allclose(idx, [[0], [1]])

    with pytest.raises(ValueError):
        labels = [["bad-label"], ["label-2"]]
        _ = axis.coord_to_idx(coord=labels)


def test_mixed_axes():
    label_axis = LabelMapAxis(labels=["label-1", "label-2", "label-3"], name="label")

    time_axis = TimeMapAxis(
        edges_min=[1, 10] * u.day,
        edges_max=[2, 13] * u.day,
        reference_time=Time("2020-03-19"),
    )

    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=4)

    axes = MapAxes(axes=[energy_axis, time_axis, label_axis])

    coords = axes.get_coord()

    assert coords["label"].shape == (1, 1, 3)
    assert coords["energy"].shape == (4, 1, 1)
    assert coords["time"].shape == (1, 2, 1)

    idx = axes.coord_to_idx(coords)

    assert_allclose(idx[0], np.arange(4).reshape((4, 1, 1)))
    assert_allclose(idx[1], np.arange(2).reshape((1, 2, 1)))
    assert_allclose(idx[2], np.arange(3).reshape((1, 1, 3)))

    hdu = axes.to_table_hdu(format="gadf")

    table = Table.read(hdu)

    assert table["LABEL"].dtype == np.dtype("U7")
    assert len(table) == 24


def test_map_axis_format_plot_xaxis():
    axis = MapAxis.from_energy_bounds(
        "0.03 TeV", "300 TeV", nbin=20, per_decade=True, name="energy_true"
    )

    with mpl_plot_check():
        ax = plt.gca()
        with quantity_support():
            ax.plot(axis.center, np.ones_like(axis.center))

    ax1 = axis.format_plot_xaxis(ax=ax)
    assert ax1.xaxis.label.properties()["text"] == "True Energy [TeV]"


def test_time_map_axis_format_plot_xaxis(time_intervals):
    axis = TimeMapAxis(
        time_intervals["t_min"],
        time_intervals["t_max"],
        time_intervals["t_ref"],
        name="time",
    )

    with mpl_plot_check():
        ax = plt.gca()
        with quantity_support():
            ax.plot(axis.center, np.ones_like(axis.center))

    ax1 = axis.format_plot_xaxis(ax=ax)
    assert ax1.xaxis.label.properties()["text"] == "Time [iso]"

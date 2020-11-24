# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.data import EventList
from gammapy.irf import EDispKernel
from gammapy.maps import Map, MapAxis, RegionGeom, RegionNDMap
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@pytest.fixture
def region_map():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=6, name="energy")
    m = Map.create(
        region="icrs;circle(83.63, 21.51, 1)",
        map_type="region",
        axes=[axis],
        unit="1/TeV",
    )
    m.data = np.arange(m.data.size, dtype=float).reshape(m.geom.data_shape)
    return m


@pytest.fixture
def region_map_true():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=6, name="energy_true")
    m = Map.create(
        region="icrs;circle(83.63, 21.51, 1)",
        map_type="region",
        axes=[axis],
        unit="1/TeV",
    )
    m.data = np.arange(m.data.size, dtype=float).reshape(m.geom.data_shape)
    return m


def test_region_nd_map(region_map):
    assert_allclose(region_map.data.sum(), 15)
    assert region_map.geom.frame == "icrs"
    assert region_map.unit == "TeV-1"
    assert region_map.data.dtype == float
    assert "RegionNDMap" in str(region_map)
    assert "1 / TeV" in str(region_map)


def test_region_nd_map_sum_over_axes(region_map):
    region_map_summed = region_map.sum_over_axes()
    weights = RegionNDMap.from_geom(region_map.geom, data=1.0)
    weights.data[5, :, :] = 0
    region_map_summed_weights = region_map.sum_over_axes(weights=weights)

    assert_allclose(region_map_summed.data, 15)
    assert_allclose(region_map_summed.data.shape, (1, 1, 1,))
    assert_allclose(region_map_summed_weights.data, 10)


@requires_dependency("matplotlib")
def test_region_nd_map_plot(region_map):
    import matplotlib.pyplot as plt

    with mpl_plot_check():
        region_map.plot()

    ax = plt.subplot(projection=region_map.geom.wcs)
    with mpl_plot_check():
        region_map.plot_region(ax=ax)


def test_region_nd_map_misc(region_map):
    assert_allclose(region_map.sum_over_axes(), 15)

    stacked = region_map.copy()
    stacked.stack(region_map)
    assert_allclose(stacked.data.sum(), 30)

    stacked = region_map.copy()
    weights = Map.from_geom(region_map.geom, dtype=np.int)
    stacked.stack(region_map, weights=weights)
    assert_allclose(stacked.data.sum(), 15)


def test_stack_differen_unit():
    region = "icrs;circle(0, 0, 1)"
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)
    region_map = RegionNDMap.create(axes=[axis], unit="m2 s", region=region)
    region_map.data += 1

    region_map_other = RegionNDMap.create(axes=[axis], unit="cm2 s", region=region)
    region_map_other.data += 1

    region_map.stack(region_map_other)
    assert_allclose(region_map.data, 1.0001)


def test_region_nd_map_sample(region_map):
    upsampled = region_map.upsample(factor=2)
    assert_allclose(upsampled.data.sum(), 15)
    assert upsampled.data.shape == (12, 1, 1)

    upsampled = region_map.upsample(factor=2, preserve_counts=False)
    assert_allclose(upsampled.data[3, 0, 0], 1.25)
    assert upsampled.data.shape == (12, 1, 1)

    downsampled = region_map.downsample(factor=2)
    assert_allclose(downsampled.data.sum(), 15)
    assert_allclose(downsampled.data[2, 0, 0], 9)
    assert downsampled.data.shape == (3, 1, 1)

    downsampled = region_map.downsample(factor=2, preserve_counts=False)
    assert_allclose(downsampled.data.sum(), 7.5)
    assert_allclose(downsampled.data[2, 0, 0], 4.5)
    assert downsampled.data.shape == (3, 1, 1)


def test_region_nd_map_get(region_map):
    values = region_map.get_by_idx((0, 0, [2, 3]))
    assert_allclose(values.squeeze(), [2, 3])

    values = region_map.get_by_pix((0, 0, [2.3, 3.7]))
    assert_allclose(values.squeeze(), [2, 4])

    energies = region_map.geom.axes[0].center
    values = region_map.get_by_coord((83.63, 21.51, energies[[0, -1]]))
    assert_allclose(values.squeeze(), [0, 5])


def test_region_nd_map_set(region_map):
    region_map = region_map.copy()
    region_map.set_by_idx((0, 0, [2, 3]), [42, 42])
    assert_allclose(region_map.data[[2, 3]], 42)

    region_map = region_map.copy()
    region_map.set_by_pix((0, 0, [2.3, 3.7]), [42, 42])
    assert_allclose(region_map.data[[2, 3]], 42)

    region_map = region_map.copy()
    energies = region_map.geom.axes[0].center
    region_map.set_by_coord((83.63, 21.51, energies[[0, -1]]), [42, 42])
    assert_allclose(region_map.data[[0, -1]], 42)


@requires_data()
def test_region_nd_map_fill_events(region_map):
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    events = EventList.read(filename)
    region_map = Map.from_geom(region_map.geom)
    region_map.fill_events(events)

    assert_allclose(region_map.data.sum(), 665)


def test_apply_edisp(region_map_true):
    e_true = region_map_true.geom.axes[0].edges
    e_reco = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3).edges

    edisp = EDispKernel.from_diagonal_response(energy_true=e_true, energy=e_reco)

    m = region_map_true.apply_edisp(edisp)
    assert m.geom.data_shape == (3, 1, 1)

    e_reco = m.geom.axes[0].edges
    assert e_reco.unit == "TeV"
    assert m.geom.axes[0].name == "energy"
    assert_allclose(e_reco[[0, -1]].value, [1, 10])


def test_regionndmap_resample_axis():
    axis_1 = MapAxis.from_edges([1, 2, 3, 4, 5], name="test-1")
    axis_2 = MapAxis.from_edges([1, 2, 3, 4], name="test-2")

    geom = RegionGeom.create(
        region="icrs;circle(83.63, 21.51, 1)", axes=[axis_1, axis_2]
    )
    m = RegionNDMap(geom, unit="m2")
    m.data += 1

    new_axis = MapAxis.from_edges([2, 3, 5], name="test-1")
    m2 = m.resample_axis(axis=new_axis)
    assert m2.data.shape == (3, 2, 1, 1)
    assert_allclose(m2.data[0, :, 0, 0], [1, 2])

    # Test without all interval covered
    new_axis = MapAxis.from_edges([1.7, 4], name="test-1")
    m3 = m.resample_axis(axis=new_axis)
    assert m3.data.shape == (3, 1, 1, 1)
    assert_allclose(m3.data, 2)

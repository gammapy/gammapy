# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.time import Time
from regions import CircleSkyRegion
import matplotlib.pyplot as plt
from gammapy.data import EventList
from gammapy.irf import EDispKernel
from gammapy.maps import (
    LabelMapAxis,
    Map,
    MapAxis,
    RegionGeom,
    RegionNDMap,
    TimeMapAxis,
)
from gammapy.utils.testing import mpl_plot_check, requires_data


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
def region_map_no_region():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=6, name="energy")
    m = Map.create(
        region=None,
        map_type="region",
        axes=[axis],
        unit="1/TeV",
    )
    m.data = np.arange(m.data.size, dtype=float).reshape(m.geom.data_shape)
    return m


@pytest.fixture
def point_region_map():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=6, name="energy")
    m = Map.create(
        region="icrs;point(83.63, 21.51)",
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
    assert_allclose(
        region_map_summed.data.shape,
        (
            1,
            1,
            1,
        ),
    )
    assert_allclose(region_map_summed_weights.data, 10)


def test_region_nd_map_plot(region_map):
    with mpl_plot_check():
        region_map.plot()

    ax = plt.subplot(projection=region_map.geom.wcs)
    with mpl_plot_check():
        region_map.plot_region(ax=ax)


def test_region_nd_map_plot_two_axes():
    energy_axis = MapAxis.from_energy_edges([1, 3, 10] * u.TeV)

    time_ref = Time("1999-01-01T00:00:00.123456789")

    time_axis = TimeMapAxis(
        edges_min=[0, 1, 3] * u.d,
        edges_max=[0.8, 1.9, 5.4] * u.d,
        reference_time=time_ref,
    )

    m = RegionNDMap.create("icrs;circle(0, 0, 1)", axes=[energy_axis, time_axis])
    m.data = 10 + np.random.random(m.data.size)

    with mpl_plot_check():
        m.plot(axis_name="energy")

    with mpl_plot_check():
        m.plot(axis_name="time")

    with pytest.raises(ValueError):
        m.plot()


def test_region_nd_map_plot_label_axis():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=5)
    label_axis = LabelMapAxis(labels=["dataset-1", "dataset-2"], name="dataset")

    m = RegionNDMap.create(region=None, axes=[energy_axis, label_axis])

    with mpl_plot_check():
        m.plot(axis_name="energy")

    with mpl_plot_check():
        m.plot(axis_name="dataset")


def test_label_axis_io(tmpdir):
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=5)
    label_axis = LabelMapAxis(labels=["dataset-1", "dataset-2"], name="dataset")

    m = RegionNDMap.create(region=None, axes=[energy_axis, label_axis])
    m.data = np.arange(m.data.size)

    filename = tmpdir / "test.fits"

    m.write(filename, format="gadf")

    m_new = RegionNDMap.read(filename, format="gadf")

    assert m.geom.axes["dataset"] == m_new.geom.axes["dataset"]
    assert m.geom.axes["energy"] == m_new.geom.axes["energy"]


def test_region_plot_mask(region_map):
    mask = region_map.geom.energy_mask(2.5 * u.TeV, 6 * u.TeV)
    with mpl_plot_check():
        mask.plot_mask()


def test_region_nd_map_misc(region_map):
    assert_allclose(region_map.sum_over_axes(), 15)

    stacked = region_map.copy()
    stacked.stack(region_map)
    assert_allclose(stacked.data.sum(), 30)

    stacked = region_map.copy()
    weights = Map.from_geom(region_map.geom, dtype=np.int)
    stacked.stack(region_map, weights=weights)
    assert_allclose(stacked.data.sum(), 15)


def test_stack_different_unit():
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


def test_region_nd_map_get_no_region(region_map_no_region):
    energies = region_map_no_region.geom.axes[0].center
    values = region_map_no_region.get_by_coord((energies[[0, -1]],))
    assert_allclose(values.squeeze(), [0, 5])

    values = region_map_no_region.get_by_coord({"energy": energies[[0, -1]]})
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


@requires_data()
def test_region_nd_map_fill_events_point_sky_region(point_region_map):
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    events = EventList.read(filename).select_offset([70.0, 71] * u.deg)
    region_map = Map.from_geom(point_region_map.geom)
    region_map.fill_events(events)

    assert_allclose(region_map.data.sum(), 0)


def test_apply_edisp(region_map_true):
    e_true = region_map_true.geom.axes[0]
    e_reco = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)

    edisp = EDispKernel.from_diagonal_response(
        energy_axis_true=e_true, energy_axis=e_reco
    )

    m = region_map_true.apply_edisp(edisp)
    assert m.geom.data_shape == (3, 1, 1)

    e_reco = m.geom.axes[0].edges
    assert e_reco.unit == "TeV"
    assert m.geom.axes[0].name == "energy"
    assert_allclose(e_reco[[0, -1]].value, [1, 10])


def test_region_nd_map_resample_axis():
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


def test_region_nd_io_ogip(tmpdir):
    energy_axis = MapAxis.from_energy_bounds(0.1, 10, 12, unit="TeV")
    m = RegionNDMap.create(
        "icrs;circle(83.63, 22.01, 0.5)", axes=[energy_axis], binsz_wcs="0.01deg"
    )
    m.write(tmpdir / "test.fits", format="ogip")

    m_new = RegionNDMap.read(tmpdir / "test.fits", format="ogip")

    assert isinstance(m_new.geom.region, CircleSkyRegion)

    geom = m_new.geom.to_wcs_geom()
    assert geom.data_shape == (12, 102, 102)

    with pytest.raises(ValueError):
        m.write(tmpdir / "test.fits", format="ogip-arf")


def test_region_nd_io_ogip_arf(tmpdir):
    energy_axis = MapAxis.from_energy_bounds(
        0.1, 10, 12, unit="TeV", name="energy_true"
    )
    m = RegionNDMap.create("icrs;circle(83.63, 22.01, 0.5)", axes=[energy_axis])
    m.write(tmpdir / "test.fits", format="ogip-arf")

    m_new = RegionNDMap.read(tmpdir / "test.fits", format="ogip-arf")

    assert m_new.geom.region is None

    with pytest.raises(ValueError):
        m.write(tmpdir / "test.fits", format="ogip")


def test_region_nd_io_gadf(tmpdir):
    energy_axis = MapAxis.from_edges([1, 3, 10] * u.TeV, name="energy")
    m = RegionNDMap.create("icrs;circle(83.63, 22.01, 0.5)", axes=[energy_axis])
    m.write(tmpdir / "test.fits", format="gadf")

    m_new = RegionNDMap.read(tmpdir / "test.fits", format="gadf")

    assert isinstance(m_new.geom.region, CircleSkyRegion)
    assert m_new.geom.axes[0].name == "energy"
    assert m_new.data.shape == (2, 1, 1)
    assert_allclose(m_new.geom.axes["energy"].edges, [1, 3, 10] * u.TeV)


def test_region_nd_io_gadf_no_region(tmpdir):
    energy_axis = MapAxis.from_edges([1, 3, 10] * u.TeV, name="energy")
    m = RegionNDMap.create(region=None, axes=[energy_axis])
    m.write(tmpdir / "test.fits", format="gadf", hdu="TEST")

    m_new = RegionNDMap.read(tmpdir / "test.fits", format="gadf", hdu="TEST")

    assert m_new.geom.region is None
    assert m_new.geom.axes[0].name == "energy"
    assert m_new.data.shape == (2, 1, 1)
    assert_allclose(m_new.geom.axes["energy"].edges, [1, 3, 10] * u.TeV)


def test_region_nd_io_gadf_rad_axis(tmpdir):
    energy_axis = MapAxis.from_edges([1, 3, 10] * u.TeV, name="energy")
    rad_axis = MapAxis.from_nodes([0, 0.1, 0.2] * u.deg, name="rad")
    m = RegionNDMap.create(
        "icrs;circle(83.63, 22.01, 0.5)", axes=[energy_axis, rad_axis], unit="sr-1"
    )
    m.data = np.arange(np.prod(m.data.shape)).reshape(m.data.shape)

    m.write(tmpdir / "test.fits", format="gadf")

    m_new = RegionNDMap.read(tmpdir / "test.fits", format="gadf")

    assert isinstance(m_new.geom.region, CircleSkyRegion)
    assert m_new.geom.axes.names == ["energy", "rad"]
    assert m_new.unit == "sr-1"

    # check that the data is not re-shuffled
    assert_allclose(m_new.data, m.data)
    assert m_new.data.shape == (3, 2, 1, 1)


def test_region_nd_hdulist():
    energy_axis = MapAxis.from_edges([1, 3, 10] * u.TeV, name="energy")
    m = RegionNDMap.create(region="icrs;circle(83.63, 22.01, 0.5)", axes=[energy_axis])

    hdulist = m.to_hdulist()
    assert hdulist[0].name == "PRIMARY"
    assert hdulist[1].name == "SKYMAP"
    assert hdulist[2].name == "SKYMAP_BANDS"
    assert hdulist[3].name == "SKYMAP_REGION"


def test_region_nd_map_interp_no_region():
    energy_axis = MapAxis.from_energy_edges([1, 3, 10] * u.TeV)

    time_ref = Time("1999-01-01T00:00:00.123456789")

    time_axis = TimeMapAxis(
        edges_min=[0, 1, 3] * u.d,
        edges_max=[0.8, 1.9, 5.4] * u.d,
        reference_time=time_ref,
    )

    m = RegionNDMap.create(region=None, axes=[energy_axis, time_axis])
    m.data = np.arange(6).reshape((energy_axis.nbin, time_axis.nbin))

    energy = [2, 6] * u.TeV
    time = time_ref + [[0.4], [1.5], [4.2]] * u.d

    value = m.interp_by_coord({"energy": energy, "time": time})
    reference = np.array(
        [
            [0.13093, 1.075717],
            [2.242041, 3.186828],
            [4.13093, 5.075717],
        ]
    )
    assert_allclose(value, reference, rtol=1e-5)

    value = m.interp_by_coord((energy, time))
    assert_allclose(value, reference, rtol=1e-5)


def test_region_map_sampling(region_map):
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=6, name="energy")
    edges = np.linspace(0.0, 30.0, 3) * u.min
    time_axis = MapAxis.from_edges(edges, name="time")

    npred_map = Map.create(
        region="icrs;circle(83.63, 21.51, 1)",
        map_type="region",
        axes=[time_axis, energy_axis],
    )
    npred_map.data[...] = 5

    coords = npred_map.sample_coord(n_events=2, random_state=0)

    assert len(coords["lon"]) == 2
    assert_allclose(coords["lon"].data, [83.63, 83.63], rtol=1e-5)
    assert_allclose(coords["lat"].data, [21.51, 21.51], rtol=1e-5)
    assert_allclose(coords["energy"].data, [3.985296, 5.721113], rtol=1e-5)
    assert_allclose(coords["time"].data, [6.354822, 9.688412], rtol=1e-5)

    assert coords["lon"].unit == "deg"
    assert coords["lat"].unit == "deg"
    assert coords["energy"].unit == "TeV"
    assert coords["time"].unit == "min"

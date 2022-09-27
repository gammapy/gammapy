# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from gammapy.data import DataStore
from gammapy.datasets import MapDataset
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def observations():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_id = [110380, 111140]
    return data_store.get_observations(obs_id)


@pytest.fixture
def observations_hess_dl3():
    """HESS DL3 observation list."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids = [23523, 23526]
    return datastore.get_observations(obs_ids)


@pytest.fixture(scope="session")
def observation_cta_1dc():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    return data_store.obs(110380)


@pytest.fixture(scope="session")
def dataset(observation_cta_1dc):
    axis = MapAxis.from_bounds(
        0.1, 10, nbin=16, unit="TeV", name="energy", interp="log"
    )
    axis_true = MapAxis.from_bounds(
        0.1, 50, nbin=30, unit="TeV", name="energy_true", interp="log"
    )
    geom = WcsGeom.create(
        npix=(11, 11), axes=[axis], skydir=observation_cta_1dc.pointing_radec
    )

    empty_dataset = MapDataset.create(geom=geom, energy_axis_true=axis_true)
    dataset_maker = MapDatasetMaker()
    return dataset_maker.run(dataset=empty_dataset, observation=observation_cta_1dc)


@pytest.fixture(scope="session")
def shifted_dataset(observation_cta_1dc):
    axis = MapAxis.from_bounds(0.1, 1, nbin=5, unit="TeV", name="energy", interp="log")
    axis_true = MapAxis.from_bounds(
        0.1, 2, nbin=10, unit="TeV", name="energy_true", interp="log"
    )
    skydir = observation_cta_1dc.pointing_radec.directional_offset_by(
        position_angle=0.0 * u.deg, separation=10 * u.deg
    )
    geom = WcsGeom.create(npix=(11, 11), axes=[axis], skydir=skydir)

    empty_dataset = MapDataset.create(
        geom=geom, energy_axis_true=axis_true, name="shifted"
    )
    dataset_maker = MapDatasetMaker()
    return dataset_maker.run(dataset=empty_dataset, observation=observation_cta_1dc)


@requires_data()
def test_safe_mask_maker_offset_max(dataset, observation_cta_1dc):
    safe_mask_maker = SafeMaskMaker(
        offset_max="3 deg", position=observation_cta_1dc.pointing_radec
    )

    mask_offset = safe_mask_maker.make_mask_offset_max(
        dataset=dataset, observation=observation_cta_1dc
    )
    assert_allclose(mask_offset.sum(), 109)


@requires_data()
def test_safe_mask_maker_aeff_default(dataset, observation_cta_1dc, caplog):
    safe_mask_maker = SafeMaskMaker(position=observation_cta_1dc.pointing_radec)

    mask_energy_aeff_default = safe_mask_maker.make_mask_energy_aeff_default(
        dataset=dataset, observation=observation_cta_1dc
    )
    assert_allclose(mask_energy_aeff_default.data.sum(), 1936)

    assert "WARNING" in [_.levelname for _ in caplog.records]
    messages = [_.message for _ in caplog.records]

    message = "No default upper safe energy threshold defined for obs 110380"
    assert message == messages[0]

    message = "No default lower safe energy threshold defined for obs 110380"
    assert message == messages[1]


@requires_data()
def test_safe_mask_maker_aeff_max(dataset, observation_cta_1dc):
    safe_mask_maker = SafeMaskMaker(position=observation_cta_1dc.pointing_radec)

    mask_aeff_max = safe_mask_maker.make_mask_energy_aeff_max(dataset)

    assert_allclose(mask_aeff_max.data.sum(), 1210)


@requires_data()
def test_safe_mask_maker_aeff_max_fixed_observation(
    dataset, shifted_dataset, observation_cta_1dc, caplog
):
    safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=20)

    mask_aeff_max = safe_mask_maker.make_mask_energy_aeff_max(
        dataset, observation=observation_cta_1dc
    )
    assert_allclose(mask_aeff_max.data.sum(), 847)

    with caplog.at_level(logging.WARNING):
        mask_aeff_max_bis = safe_mask_maker.make_mask_energy_aeff_max(
            shifted_dataset, observation=observation_cta_1dc
        )

    assert len(caplog.record_tuples) == 1
    assert caplog.record_tuples[0] == (
        "gammapy.makers.safe",
        logging.WARNING,
        "Effective area is all zero at [267d40m52.368168s -19d36m27s]. No safe "
        "energy band can be defined for the dataset 'shifted': setting `mask_safe`"
        " to all False.",
    )
    assert_allclose(mask_aeff_max_bis.data.sum(), 0)


@requires_data()
def test_safe_mask_maker_aeff_max_fixed_offset(dataset, observation_cta_1dc):
    safe_mask_maker = SafeMaskMaker(
        methods=["aeff-max"], aeff_percent=20, fixed_offset=5 * u.deg
    )

    mask_aeff_max = safe_mask_maker.make_mask_energy_aeff_max(
        dataset, observation=observation_cta_1dc
    )
    assert_allclose(mask_aeff_max.data.sum(), 726)

    with pytest.raises(ValueError):
        mask_aeff_max = safe_mask_maker.make_mask_energy_aeff_max(dataset)


@requires_data()
def test_safe_mask_maker_offset_max_fixed_offset(dataset, observation_cta_1dc):
    safe_mask_maker_offset = SafeMaskMaker(offset_max="3 deg", fixed_offset=1.5 * u.deg)

    mask_aeff_max_offset = safe_mask_maker_offset.make_mask_energy_aeff_max(
        dataset=dataset, observation=observation_cta_1dc
    )
    assert_allclose(mask_aeff_max_offset.data.sum(), 1210)


@requires_data()
def test_safe_mask_maker_edisp_bias(dataset, observation_cta_1dc):
    safe_mask_maker = SafeMaskMaker(
        bias_percent=0.02, position=observation_cta_1dc.pointing_radec
    )

    mask_edisp_bias = safe_mask_maker.make_mask_energy_edisp_bias(dataset=dataset)
    assert_allclose(mask_edisp_bias.data.sum(), 1815)


@requires_data()
def test_safe_mask_maker_edisp_bias_fixed_offset(dataset, observation_cta_1dc):
    safe_mask_maker_offset = SafeMaskMaker(
        offset_max="3 deg", bias_percent=0.02, fixed_offset=1.5 * u.deg
    )

    mask_edisp_bias_offset = safe_mask_maker_offset.make_mask_energy_edisp_bias(
        dataset=dataset, observation=observation_cta_1dc
    )
    assert_allclose(mask_edisp_bias_offset.data.sum(), 1694)


@requires_data()
def test_safe_mask_maker_bkg_peak(dataset, observation_cta_1dc):
    safe_mask_maker = SafeMaskMaker(position=observation_cta_1dc.pointing_radec)

    mask_bkg_peak = safe_mask_maker.make_mask_energy_bkg_peak(dataset)
    assert_allclose(mask_bkg_peak.data.sum(), 1936)


@requires_data()
def test_safe_mask_maker_bkg_peak_first_bin(dataset, observation_cta_1dc):
    safe_mask_maker = SafeMaskMaker(position=observation_cta_1dc.pointing_radec)

    dataset_maker = MapDatasetMaker()

    axis = MapAxis.from_bounds(1.0, 10, nbin=6, unit="TeV", name="energy", interp="log")

    geom = WcsGeom.create(
        npix=(5, 5), axes=[axis], skydir=observation_cta_1dc.pointing_radec
    )
    empty_dataset = MapDataset.create(geom=geom)
    dataset = dataset_maker.run(empty_dataset, observation_cta_1dc)
    mask_bkg_peak = safe_mask_maker.make_mask_energy_bkg_peak(dataset)
    assert np.all(mask_bkg_peak)


@requires_data()
def test_safe_mask_maker_no_root(dataset, observation_cta_1dc):
    safe_mask_maker_noroot = SafeMaskMaker(
        offset_max="3 deg", aeff_percent=-10, bias_percent=-10
    )
    mask_aeff_max_noroot = safe_mask_maker_noroot.make_mask_energy_aeff_max(dataset)
    mask_edisp_bias_noroot = safe_mask_maker_noroot.make_mask_energy_edisp_bias(dataset)
    assert_allclose(mask_aeff_max_noroot.data.sum(), 1815)
    assert_allclose(mask_edisp_bias_noroot.data.sum(), 1936)


@requires_data()
def test_safe_mask_maker_bkg_invalid(observations_hess_dl3):
    obs = observations_hess_dl3[0]

    axis = MapAxis.from_bounds(
        0.1, 10, nbin=16, unit="TeV", name="energy", interp="log"
    )
    axis_true = MapAxis.from_bounds(
        0.1, 50, nbin=30, unit="TeV", name="energy_true", interp="log"
    )
    geom = WcsGeom.create(npix=(9, 9), axes=[axis], skydir=obs.pointing_radec)

    empty_dataset = MapDataset.create(geom=geom, energy_axis_true=axis_true)
    dataset_maker = MapDatasetMaker()

    safe_mask_maker_nonan = SafeMaskMaker([])

    dataset = dataset_maker.run(empty_dataset, obs)
    bkg = dataset.background.data
    bkg[0, 0, 0] = np.nan

    mask_nonan = safe_mask_maker_nonan.make_mask_bkg_invalid(dataset)

    assert not mask_nonan[0, 0, 0]

    assert_allclose(bkg[mask_nonan].max(), 20.656366)

    dataset = safe_mask_maker_nonan.run(dataset, obs)
    assert_allclose(dataset.mask_safe, mask_nonan)

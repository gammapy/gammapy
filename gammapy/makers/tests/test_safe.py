# Licensed under a 3-clause BSD style license - see LICENSE.rst
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


@requires_data()
def test_safe_mask_maker(observations, caplog):
    obs = observations[0]

    axis = MapAxis.from_bounds(
        0.1, 10, nbin=16, unit="TeV", name="energy", interp="log"
    )
    axis_true = MapAxis.from_bounds(
        0.1, 50, nbin=30, unit="TeV", name="energy_true", interp="log"
    )
    geom = WcsGeom.create(npix=(11, 11), axes=[axis], skydir=obs.pointing_radec)

    empty_dataset = MapDataset.create(geom=geom, energy_axis_true=axis_true)
    dataset_maker = MapDatasetMaker()
    safe_mask_maker = SafeMaskMaker(
        offset_max="3 deg", bias_percent=0.02, position=obs.pointing_radec
    )

    fixed_offset = 1.5 * u.deg
    safe_mask_maker_offset = SafeMaskMaker(
        offset_max="3 deg", bias_percent=0.02, fixed_offset=fixed_offset
    )

    dataset = dataset_maker.run(empty_dataset, obs)

    mask_offset = safe_mask_maker.make_mask_offset_max(dataset=dataset, observation=obs)
    assert_allclose(mask_offset.sum(), 109)

    mask_energy_aeff_default = safe_mask_maker.make_mask_energy_aeff_default(
        dataset=dataset, observation=obs
    )
    assert_allclose(mask_energy_aeff_default.data.sum(), 1936)

    mask_aeff_max = safe_mask_maker.make_mask_energy_aeff_max(dataset)
    mask_aeff_max_offset = safe_mask_maker_offset.make_mask_energy_aeff_max(
        dataset, obs
    )
    assert_allclose(mask_aeff_max.data.sum(), 1210)
    assert_allclose(mask_aeff_max_offset.data.sum(), 1210)

    mask_edisp_bias = safe_mask_maker.make_mask_energy_edisp_bias(dataset)
    mask_edisp_bias_offset = safe_mask_maker_offset.make_mask_energy_edisp_bias(
        dataset, obs
    )
    assert_allclose(mask_edisp_bias.data.sum(), 1815)
    assert_allclose(mask_edisp_bias_offset.data.sum(), 1694)

    mask_bkg_peak = safe_mask_maker.make_mask_energy_bkg_peak(dataset)
    assert_allclose(mask_bkg_peak.data.sum(), 1815)
    assert "WARNING" in [_.levelname for _ in caplog.records]
    message1 = "No default thresholds defined for obs 110380"
    assert message1 in [_.message for _ in caplog.records]

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
    geom = WcsGeom.create(npix=(11, 11), axes=[axis], skydir=obs.pointing_radec)

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

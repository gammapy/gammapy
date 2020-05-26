# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
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


@requires_data()
def test_safe_mask_maker(observations):
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

    dataset = dataset_maker.run(empty_dataset, obs)

    mask_offset = safe_mask_maker.make_mask_offset_max(dataset=dataset, observation=obs)
    assert_allclose(mask_offset.sum(), 109)

    mask_energy_aeff_default = safe_mask_maker.make_mask_energy_aeff_default(
        dataset=dataset, observation=obs
    )
    assert_allclose(mask_energy_aeff_default.sum(), 1936)

    mask_aeff_max = safe_mask_maker.make_mask_energy_aeff_max(dataset)
    assert_allclose(mask_aeff_max.sum(), 1210)

    mask_edisp_bias = safe_mask_maker.make_mask_energy_edisp_bias(dataset)
    assert_allclose(mask_edisp_bias.sum(), 1815)

    mask_bkg_peak = safe_mask_maker.make_mask_energy_bkg_peak(dataset)
    assert_allclose(mask_bkg_peak.sum(), 1815)

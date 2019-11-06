# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.maps import WcsGeom, WcsNDMap
from gammapy.spectrum import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@pytest.fixture
def observations_hess_dl3():
    """HESS DL3 observation list."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids = [23523, 23526]
    return datastore.get_observations(obs_ids)


@pytest.fixture
def observations_cta_dc1():
    """CTA DC1 observation list."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_ids = [110380, 111140]
    return datastore.get_observations(obs_ids)


@pytest.fixture()
def spectrum_dataset_maker_gc():
    pos = SkyCoord(0.0, 0.0, unit="deg", frame="galactic")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)

    e_reco = np.logspace(0, 2, 5) * u.TeV
    e_true = np.logspace(-0.5, 2, 11) * u.TeV
    return SpectrumDatasetMaker(region=region, e_reco=e_reco, e_true=e_true)


@pytest.fixture()
def spectrum_dataset_maker_crab():
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)
    e_reco = np.logspace(0, 2, 5) * u.TeV
    e_true = np.logspace(-0.5, 2, 11) * u.TeV
    return SpectrumDatasetMaker(region=region, e_reco=e_reco, e_true=e_true)


@pytest.fixture
def spectrum_dataset_maker_crab_fine_bins():
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)
    e_true = np.logspace(-2, 2.5, 109) * u.TeV
    e_reco = np.logspace(-2, 2, 73) * u.TeV
    return SpectrumDatasetMaker(region=region, e_reco=e_reco, e_true=e_true)


@pytest.fixture
def reflected_regions_bkg_maker():
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    exclusion_region = CircleSkyRegion(pos, Angle(0.3, "deg"))
    geom = WcsGeom.create(skydir=pos, binsz=0.02, width=10.0)
    mask = geom.region_mask([exclusion_region], inside=False)
    exclusion_mask = WcsNDMap(geom, data=mask)

    return ReflectedRegionsBackgroundMaker(
        exclusion_mask=exclusion_mask, min_distance_input="0.2 deg"
    )


@requires_data()
def test_spectrum_dataset_maker_hess_dl3(
    spectrum_dataset_maker_crab, observations_hess_dl3
):
    datasets = []

    for obs in observations_hess_dl3:
        dataset = spectrum_dataset_maker_crab.run(obs)
        datasets.append(dataset)

    assert_allclose(datasets[0].counts.data.sum(), 100)
    assert_allclose(datasets[1].counts.data.sum(), 92)

    assert_allclose(datasets[0].livetime.value, 1581.736758)
    assert_allclose(datasets[1].livetime.value, 1572.686724)

    assert_allclose(datasets[0].background.data.sum(), 7.74732, rtol=1e-5)
    assert_allclose(datasets[1].background.data.sum(), 6.118879, rtol=1e-5)


@requires_data()
def test_spectrum_dataset_maker_hess_cta(
    spectrum_dataset_maker_gc, observations_cta_dc1
):
    datasets = []

    for obs in observations_cta_dc1:
        dataset = spectrum_dataset_maker_gc.run(obs)
        datasets.append(dataset)

    assert_allclose(datasets[0].counts.data.sum(), 53)
    assert_allclose(datasets[1].counts.data.sum(), 47)

    assert_allclose(datasets[0].livetime.value, 1764.000034)
    assert_allclose(datasets[1].livetime.value, 1764.000034)

    assert_allclose(datasets[0].background.data.sum(), 2.238345, rtol=1e-5)
    assert_allclose(datasets[1].background.data.sum(), 2.164593, rtol=1e-5)


@requires_data()
def test_safe_mask_maker_dl3(spectrum_dataset_maker_crab, observations_hess_dl3):
    safe_mask_maker = SafeMaskMaker()

    obs = observations_hess_dl3[0]
    dataset = spectrum_dataset_maker_crab.run(obs)
    dataset = safe_mask_maker.run(dataset, obs)
    assert_allclose(dataset.energy_range[0].value, 1)
    assert dataset.energy_range[0].unit == "TeV"

    mask_safe = safe_mask_maker.make_mask_energy_aeff_max(dataset)
    assert mask_safe.sum() == 4

    mask_safe = safe_mask_maker.make_mask_energy_edisp_bias(dataset)
    assert mask_safe.sum() == 3


@requires_data()
def test_safe_mask_maker_dc1(spectrum_dataset_maker_gc, observations_cta_dc1):
    safe_mask_maker = SafeMaskMaker(methods=["edisp-bias", "aeff-max"])

    obs = observations_cta_dc1[0]
    dataset = spectrum_dataset_maker_gc.run(obs)
    dataset = safe_mask_maker.run(dataset, obs)
    assert_allclose(dataset.energy_range[0].value, 3.162278, rtol=1e-3)
    assert dataset.energy_range[0].unit == "TeV"


@requires_data()
class TestSpectrumMakerChain:
    @staticmethod
    @pytest.mark.parametrize(
        "pars, results",
        [
            (
                dict(containment_correction=False),
                dict(n_on=192, sigma=21.1404, aeff=580254.9 * u.m ** 2, edisp=0.236176),
            ),
            (
                dict(containment_correction=True),
                dict(
                    n_on=192,
                    sigma=21.1404,
                    aeff=361924.746081 * u.m ** 2,
                    edisp=0.236176,
                ),
            ),
        ],
    )
    def test_extract(
        pars,
        results,
        observations_hess_dl3,
        spectrum_dataset_maker_crab_fine_bins,
        reflected_regions_bkg_maker,
    ):
        """Test quantitative output for various configs"""
        safe_mask_maker = SafeMaskMaker()

        obs = observations_hess_dl3[0]
        spectrum_dataset_maker_crab_fine_bins.containment_correction = pars[
            "containment_correction"
        ]
        dataset = spectrum_dataset_maker_crab_fine_bins.run(
            obs, selection=["counts", "aeff", "edisp"]
        )
        dataset = reflected_regions_bkg_maker.run(dataset, obs)
        dataset = safe_mask_maker.run(dataset, obs)

        aeff_actual = dataset.aeff.data.evaluate(energy=5 * u.TeV)
        edisp_actual = dataset.edisp.data.evaluate(e_true=5 * u.TeV, e_reco=5.2 * u.TeV)

        assert_quantity_allclose(aeff_actual, results["aeff"], rtol=1e-3)
        assert_quantity_allclose(edisp_actual, results["edisp"], rtol=1e-3)

        # TODO: Introduce assert_stats_allclose
        info = dataset.info_dict()

        assert info["n_on"] == results["n_on"]
        assert_allclose(info["significance"], results["sigma"], atol=1e-2)

        gti_obs = obs.gti.table
        gti_dataset = dataset.gti.table
        assert_allclose(gti_dataset["START"], gti_obs["START"])
        assert_allclose(gti_dataset["STOP"], gti_obs["STOP"])

    def test_compute_energy_threshold(
        self, spectrum_dataset_maker_crab_fine_bins, observations_hess_dl3
    ):
        safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

        obs = observations_hess_dl3[0]
        spectrum_dataset_maker_crab_fine_bins.containment_correction = True
        dataset = spectrum_dataset_maker_crab_fine_bins.run(
            obs, selection=["counts", "aeff", "edisp"]
        )
        dataset = safe_mask_maker.run(dataset, obs)

        actual = dataset.energy_range[0]
        assert_quantity_allclose(actual, 0.8799225 * u.TeV, rtol=1e-3)

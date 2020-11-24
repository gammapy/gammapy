# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import SpectrumDataset
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap
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
def spectrum_dataset_gc():
    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_edges(np.logspace(-1, 2, 13) * u.TeV, name="energy_true")
    pos = SkyCoord(0.0, 0.0, unit="deg", frame="galactic")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)
    return SpectrumDataset.create(e_reco, e_true, region=region)


@pytest.fixture()
def spectrum_dataset_crab():
    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_edges(np.logspace(-0.5, 2, 11) * u.TeV, name="energy_true")
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)
    return SpectrumDataset.create(e_reco, e_true, region=region)


@pytest.fixture()
def spectrum_dataset_crab_fine():
    e_true = MapAxis.from_edges(np.logspace(-2, 2.5, 109) * u.TeV, name="energy_true")
    e_reco = MapAxis.from_edges(np.logspace(-2, 2, 73) * u.TeV, name="energy")
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)
    return SpectrumDataset.create(e_reco, e_true, region=region)


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
def test_spectrum_dataset_maker_hess_dl3(spectrum_dataset_crab, observations_hess_dl3):
    datasets = []
    maker = SpectrumDatasetMaker()

    for obs in observations_hess_dl3:
        dataset = maker.run(spectrum_dataset_crab, obs)
        datasets.append(dataset)

    assert_allclose(datasets[0].counts.data.sum(), 100)
    assert_allclose(datasets[1].counts.data.sum(), 92)

    assert_allclose(datasets[0].exposure.meta["livetime"].value, 1581.736758)
    assert_allclose(datasets[1].exposure.meta["livetime"].value, 1572.686724)

    assert_allclose(datasets[0].npred_background().data.sum(), 7.74732, rtol=1e-5)
    assert_allclose(datasets[1].npred_background().data.sum(), 6.118879, rtol=1e-5)


@requires_data()
def test_spectrum_dataset_maker_hess_cta(spectrum_dataset_gc, observations_cta_dc1):
    maker = SpectrumDatasetMaker()

    datasets = []

    for obs in observations_cta_dc1:
        dataset = maker.run(spectrum_dataset_gc, obs)
        datasets.append(dataset)

    assert_allclose(datasets[0].counts.data.sum(), 53)
    assert_allclose(datasets[1].counts.data.sum(), 47)

    assert_allclose(datasets[0].exposure.meta["livetime"].value, 1764.000034)
    assert_allclose(datasets[1].exposure.meta["livetime"].value, 1764.000034)

    assert_allclose(datasets[0].npred_background().data.sum(), 2.238345, rtol=1e-5)
    assert_allclose(datasets[1].npred_background().data.sum(), 2.164593, rtol=1e-5)


@requires_data()
def test_safe_mask_maker_dl3(spectrum_dataset_crab, observations_hess_dl3):

    safe_mask_maker = SafeMaskMaker()
    maker = SpectrumDatasetMaker()

    obs = observations_hess_dl3[0]
    dataset = maker.run(spectrum_dataset_crab, obs)
    dataset = safe_mask_maker.run(dataset, obs)
    assert_allclose(dataset.energy_range[0].value, 1)
    assert dataset.energy_range[0].unit == "TeV"

    mask_safe = safe_mask_maker.make_mask_energy_aeff_max(dataset)
    assert mask_safe.sum() == 4

    mask_safe = safe_mask_maker.make_mask_energy_edisp_bias(dataset)
    assert mask_safe.sum() == 3

    mask_safe = safe_mask_maker.make_mask_energy_bkg_peak(dataset)
    assert mask_safe.sum() == 3


@requires_data()
def test_safe_mask_maker_dc1(spectrum_dataset_gc, observations_cta_dc1):
    safe_mask_maker = SafeMaskMaker(methods=["edisp-bias", "aeff-max"])

    obs = observations_cta_dc1[0]
    maker = SpectrumDatasetMaker()
    dataset = maker.run(spectrum_dataset_gc, obs)
    dataset = safe_mask_maker.run(dataset, obs)
    assert_allclose(dataset.energy_range[0].value, 3.162278, rtol=1e-3)
    assert dataset.energy_range[0].unit == "TeV"


@requires_data()
def test_make_meta_table(observations_hess_dl3):
    maker_obs = SpectrumDatasetMaker()
    map_spectrumdataset_meta_table = maker_obs.make_meta_table(
        observation=observations_hess_dl3[0]
    )

    assert_allclose(map_spectrumdataset_meta_table["RA_PNT"], 83.63333129882812)
    assert_allclose(map_spectrumdataset_meta_table["DEC_PNT"], 21.51444435119629)
    assert_allclose(map_spectrumdataset_meta_table["OBS_ID"], 23523)


@requires_data()
class TestSpectrumMakerChain:
    @staticmethod
    @pytest.mark.parametrize(
        "pars, results",
        [
            (
                dict(containment_correction=False),
                dict(
                    n_on=125, sigma=18.953014, aeff=580254.9 * u.m ** 2, edisp=0.235864
                ),
            ),
            (
                dict(containment_correction=True),
                dict(
                    n_on=125,
                    sigma=18.953014,
                    aeff=375314.356461 * u.m ** 2,
                    edisp=0.235864,
                ),
            ),
        ],
    )
    def test_extract(
        pars,
        results,
        observations_hess_dl3,
        spectrum_dataset_crab_fine,
        reflected_regions_bkg_maker,
    ):
        """Test quantitative output for various configs"""
        safe_mask_maker = SafeMaskMaker()
        maker = SpectrumDatasetMaker(
            containment_correction=pars["containment_correction"]
        )

        obs = observations_hess_dl3[0]
        dataset = maker.run(spectrum_dataset_crab_fine, obs)
        dataset = reflected_regions_bkg_maker.run(dataset, obs)
        dataset = safe_mask_maker.run(dataset, obs)

        exposure_actual = (
            dataset.exposure.interp_by_coord(
                {
                    "energy_true": 5 * u.TeV,
                    "skycoord": dataset.counts.geom.center_skydir,
                }
            )
            * dataset.exposure.unit
        )

        edisp_actual = dataset.edisp.get_edisp_kernel().data.evaluate(
            energy_true=5 * u.TeV, energy=5.2 * u.TeV
        )
        aeff_actual = exposure_actual / dataset.exposure.meta["livetime"]

        assert_quantity_allclose(aeff_actual, results["aeff"], rtol=1e-3)
        assert_quantity_allclose(edisp_actual, results["edisp"], rtol=1e-3)

        # TODO: Introduce assert_stats_allclose
        info = dataset.info_dict()

        assert info["counts"] == results["n_on"]
        assert_allclose(info["sqrt_ts"], results["sigma"], rtol=1e-2)

        gti_obs = obs.gti.table
        gti_dataset = dataset.gti.table
        assert_allclose(gti_dataset["START"], gti_obs["START"])
        assert_allclose(gti_dataset["STOP"], gti_obs["STOP"])

    def test_compute_energy_threshold(
        self, spectrum_dataset_crab_fine, observations_hess_dl3
    ):

        maker = SpectrumDatasetMaker(containment_correction=True)
        safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

        obs = observations_hess_dl3[0]
        dataset = maker.run(spectrum_dataset_crab_fine, obs)
        dataset = safe_mask_maker.run(dataset, obs)

        actual = dataset.energy_range[0]
        assert_quantity_allclose(actual, 0.8799225 * u.TeV, rtol=1e-3)

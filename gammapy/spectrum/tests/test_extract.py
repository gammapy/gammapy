# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore, ObservationStats
from gammapy.maps import WcsGeom, WcsNDMap
from gammapy.spectrum import (
    SpectrumDatasetMaker,
    SafeMaskMaker,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.utils.testing import (
    assert_quantity_allclose,
    requires_data,
)


@pytest.fixture(scope="session")
def exclusion_mask():
    """Example mask for testing."""
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    exclusion_region = CircleSkyRegion(pos, Angle(0.3, "deg"))
    geom = WcsGeom.create(skydir=pos, binsz=0.02, width=10.0)
    mask = geom.region_mask([exclusion_region], inside=False)
    return WcsNDMap(geom, data=mask)


@pytest.fixture(scope="session")
def on_region():
    """Example on_region for testing."""
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)
    return region


@pytest.fixture
def observations():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs_ids = [23523, 23526]
    return datastore.get_observations(obs_ids)


@pytest.fixture
def dataset_maker(on_region):
    return SpectrumDatasetMaker(
        region=on_region,
        e_true=np.logspace(-2, 2.5, 109) * u.TeV,
        e_reco=np.logspace(-2, 2, 73) * u.TeV
    )


@pytest.fixture
def bkg_maker(on_region, exclusion_mask):
    return ReflectedRegionsBackgroundMaker(
        region=on_region,
        exclusion_mask=exclusion_mask,
        min_distance_input="0.2 deg",
    )


@requires_data()
class TestSpectrumExtraction:
    @staticmethod
    @pytest.mark.parametrize(
        "pars, results",
        [
            (
                dict(containment_correction=False),
                dict(
                    n_on=192,
                    sigma=21.1404,
                    aeff=580254.9 * u.m ** 2,
                    edisp=0.236176,
                ),
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
    def test_extract(pars, results, observations, dataset_maker, bkg_maker):
        """Test quantitative output for various configs"""
        safe_mask_maker = SafeMaskMaker()

        obs = observations[0]
        dataset_maker.containment_correction = pars["containment_correction"]
        dataset = dataset_maker.run(obs, selection=["counts", "aeff", "edisp"])
        dataset = bkg_maker.run(dataset, obs)
        dataset = safe_mask_maker.run(dataset, obs)

        aeff_actual = dataset.aeff.data.evaluate(energy=5 * u.TeV)
        edisp_actual = dataset.edisp.data.evaluate(e_true=5 * u.TeV, e_reco=5.2 * u.TeV)

        assert_quantity_allclose(aeff_actual, results["aeff"], rtol=1e-3)
        assert_quantity_allclose(edisp_actual, results["edisp"], rtol=1e-3)

        # TODO: Introduce assert_stats_allclose
        info = dataset._info_dict()
        info["obs_id"] = info.pop("name")
        stats = ObservationStats(**info)
        n_on_actual = stats.n_on
        sigma_actual = stats.sigma

        assert n_on_actual == results["n_on"]
        assert_allclose(sigma_actual, results["sigma"], atol=1e-2)

        gti_obs = obs.gti.table
        gti_dataset = dataset.gti.table
        assert_allclose(gti_dataset["START"], gti_obs["START"])
        assert_allclose(gti_dataset["STOP"], gti_obs["STOP"])

    def test_compute_energy_threshold(self, dataset_maker, observations):
        safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

        obs = observations[0]
        dataset_maker.containment_correction = True
        dataset = dataset_maker.run(obs, selection=["counts", "aeff", "edisp"])
        dataset = safe_mask_maker.run(dataset, obs)

        actual = dataset.energy_range[0]
        assert_quantity_allclose(actual, 0.8799225 * u.TeV, rtol=1e-3)

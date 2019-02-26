# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...spectrum import SpectrumExtraction, SpectrumObservation
from ...background import ReflectedRegionsBackgroundEstimator
from ...maps import WcsGeom, WcsNDMap
from ...data import DataStore


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
def bkg_estimate(observations, on_region, exclusion_mask):
    """An example background estimate"""
    est = ReflectedRegionsBackgroundEstimator(
        observations=observations,
        on_region=on_region,
        exclusion_mask=exclusion_mask,
        min_distance_input="0.2 deg",
    )
    est.run()
    return est.result


@pytest.fixture
def extraction(bkg_estimate, observations):
    """An example SpectrumExtraction for tests."""
    # Restrict true energy range covered by HAP exporter
    e_true = np.logspace(-1, 1.9, 70) * u.TeV

    return SpectrumExtraction(
        bkg_estimate=bkg_estimate, observations=observations, e_true=e_true
    )


@requires_data("gammapy-data")
class TestSpectrumExtraction:
    @pytest.mark.parametrize(
        "pars, results",
        [
            (
                dict(containment_correction=False),
                dict(
                    n_on=192,
                    sigma=20.941125,
                    aeff=580254.9 * u.m ** 2,
                    edisp=0.236176,
                    containment=1,
                ),
            ),
            (
                dict(containment_correction=True),
                dict(
                    n_on=192,
                    sigma=20.941125,
                    aeff=373237.8 * u.m ** 2,
                    edisp=0.236176,
                    containment=0.661611,
                ),
            ),
        ],
    )
    @pytest.mark.xfail
    def test_extract(self, pars, results, observations, bkg_estimate):
        """Test quantitative output for various configs"""
        extraction = SpectrumExtraction(
            observations=observations, bkg_estimate=bkg_estimate, **pars
        )

        extraction.run()
        obs = extraction.spectrum_observations[0]
        aeff_actual = obs.aeff.data.evaluate(energy=5 * u.TeV)
        edisp_actual = obs.edisp.data.evaluate(e_true=5 * u.TeV, e_reco=5.2 * u.TeV)

        assert_quantity_allclose(aeff_actual, results["aeff"], rtol=1e-3)
        assert_quantity_allclose(edisp_actual, results["edisp"], rtol=1e-3)

        containment_actual = extraction.containment[60]

        # TODO: Introduce assert_stats_allclose
        n_on_actual = obs.total_stats.n_on
        sigma_actual = obs.total_stats.sigma

        assert n_on_actual == results["n_on"]
        assert_allclose(sigma_actual, results["sigma"], atol=1e-2)
        assert_allclose(containment_actual, results["containment"], rtol=1e-3)

    @staticmethod
    def test_alpha(observations, bkg_estimate):
        bkg_estimate[0].a_off = 0
        bkg_estimate[1].a_off = 2
        extraction = SpectrumExtraction(
            observations=observations, bkg_estimate=bkg_estimate, max_alpha=0.2
        )
        extraction.run()
        assert len(extraction.spectrum_observations) == 0

    @staticmethod
    def test_run(tmpdir, extraction):
        """Test the run method and check if files are written correctly"""
        extraction.run()
        extraction.write(outdir=tmpdir, overwrite=True)
        testobs = SpectrumObservation.read(tmpdir / "ogip_data" / "pha_obs23523.fits")
        assert_quantity_allclose(
            testobs.aeff.data.data, extraction.spectrum_observations[0].aeff.data.data
        )
        assert_quantity_allclose(
            testobs.on_vector.data.data,
            extraction.spectrum_observations[0].on_vector.data.data,
        )
        assert_quantity_allclose(
            testobs.on_vector.energy.nodes,
            extraction.spectrum_observations[0].on_vector.energy.nodes,
        )

    @requires_dependency("sherpa")
    def test_sherpa(self, tmpdir, extraction):
        """Same as above for files to be used with sherpa"""
        import sherpa.astro.ui as sau

        extraction.run()
        extraction.write(outdir=tmpdir, use_sherpa=True, overwrite=True)
        sau.load_pha(str(tmpdir / "ogip_data" / "pha_obs23523.fits"))
        arf = sau.get_arf()

        actual = arf._arf._specresp
        desired = extraction.spectrum_observations[0].aeff.data.data.value
        assert_allclose(actual, desired)

    def test_compute_energy_threshold(self, extraction):
        extraction.run()
        extraction.compute_energy_threshold(method_lo="area_max", area_percent_lo=10)
        actual = extraction.spectrum_observations[0].lo_threshold
        assert_quantity_allclose(actual, 0.8799225 * u.TeV, rtol=1e-3)

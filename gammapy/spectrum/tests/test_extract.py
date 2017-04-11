# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.tests.helper import assert_quantity_allclose
from regions import CircleSkyRegion
from ...utils.testing import requires_dependency, requires_data
from ...data import DataStore, Target, ObservationList
from ...datasets import gammapy_extra
from ...image import SkyImage
from ...spectrum import SpectrumExtraction, SpectrumObservation
from ...background import ReflectedRegionsBackgroundEstimator


@pytest.fixture(scope='session')
def obs():
    """An example ObservationList for tests."""
    obs_id = [23523, 23592]
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)
    obs = ObservationList([ds.obs(_) for _ in obs_id])
    return obs


@pytest.fixture(scope='session')
def on_region():
    """An example Target for tests."""
    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.11 deg')
    on_region = CircleSkyRegion(center, radius)
    return on_region


@pytest.fixture(scope='session')
def bkg_estimator():
    """An example background_estimator"""
    exclusion_file = gammapy_extra.filename(
        "datasets/exclusion_masks/tevcat_exclusion.fits")
    excl = SkyImage.read(exclusion_file)
    bkg = ReflectedRegionsBackgroundEstimator(exclusion_mask=excl,
                                              on_region=on_region())
    return bkg


@pytest.fixture(scope='session')
def extraction():
    """An example SpectrumExtraction for tests."""
    # Restrict true energy range covered by HAP exporter
    e_true = np.logspace(-1, 1.9, 70) * u.TeV

    extraction = SpectrumExtraction(on_region=on_region(),
                                    e_true=e_true
                                    )
    return extraction


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumExtraction:
    @pytest.mark.parametrize("pars, results", [
        (dict(containment_correction=False), dict(n_on=172,
                                                  sigma=24.98,
                                                  aeff=549861.8 * u.m ** 2,
                                                  containment = 1,
                                                  edisp=0.2595896944765074
                                                 )),
        (dict(containment_correction=True), dict(n_on=172,
                                                 sigma=24.98,
                                                 aeff=549861.8 * u.m ** 2,
                                                 containment = 0.8525390663792395,
                                                 edisp=0.2595896944765074
                                                ))
    ])
    def test_extract(self, pars, results, on_region, obs, bkg_estimator, tmpdir):
        """Test quantitative output for various configs"""
        extraction = SpectrumExtraction(on_region=on_region,
                                        background_estimator=bkg_estimator,
                                        **pars)

        extraction.run(obs)
        obs = extraction.observations[0]
        aeff_actual = obs.aeff.data.evaluate(energy=5 * u.TeV)
        containment_actual = obs.on_vector.areascal[60]
        edisp_actual = obs.edisp.data.evaluate(e_true=5 * u.TeV,
                                               e_reco=5.2 * u.TeV)

        assert_quantity_allclose(aeff_actual, results['aeff'], rtol=1e-3)
        assert_quantity_allclose(edisp_actual, results['edisp'], rtol=1e-3)
        assert_allclose(containment_actual, results['containment'], rtol=1e-3)

        # TODO: Introduce assert_stats_allclose
        n_on_actual = obs.total_stats.n_on
        sigma_actual = obs.total_stats.sigma

        assert n_on_actual == results['n_on']
        assert_allclose(sigma_actual, results['sigma'], atol=1e-2)

    def test_run(self, tmpdir, extraction, obs):
        """Test the run method and check if files are written correctly"""
        extraction.run(outdir=tmpdir, obs_list=obs)
        testobs = SpectrumObservation.read(tmpdir / 'ogip_data' / 'pha_obs23523.fits')
        assert_quantity_allclose(testobs.aeff.data.data,
                                 extraction.observations[0].aeff.data.data)
        assert_quantity_allclose(testobs.on_vector.data.data,
                                 extraction.observations[0].on_vector.data.data)
        assert_quantity_allclose(testobs.on_vector.energy.nodes,
                                 extraction.observations[0].on_vector.energy.nodes)

    @requires_dependency('sherpa')
    def test_sherpa(self, tmpdir, extraction, obs):
        """Same as above for files to be used with sherpa"""
        extraction.run(outdir=tmpdir, obs_list=obs, use_sherpa=True)

        import sherpa.astro.ui as sau
        sau.load_pha(str(tmpdir / 'ogip_data' / 'pha_obs23523.fits'))
        arf = sau.get_arf()
        actual = arf._arf._specresp
        desired = extraction.observations[0].aeff.data.data.value
        assert_allclose(actual, desired)

    def test_define_energy_threshold(self, extraction):
        # TODO: Find better API for this
        extraction.define_energy_threshold(method_lo_threshold="area_max",
                                           percent=10)
        assert_quantity_allclose(extraction.observations[0].lo_threshold,
                                 0.6812920690579611 * u.TeV,
                                 rtol=1e-3)

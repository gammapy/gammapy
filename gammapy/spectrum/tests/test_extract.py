# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...spectrum import SpectrumExtraction, SpectrumObservation
from ...background.tests.test_reflected import bkg_estimator, obs_list


@pytest.fixture(scope='session')
def bkg_estimate():
    """An example background estimate"""
    est = bkg_estimator()
    est.run()
    return est.result


@pytest.fixture(scope='session')
def extraction():
    """An example SpectrumExtraction for tests."""
    # Restrict true energy range covered by HAP exporter
    e_true = np.logspace(-1, 1.9, 70) * u.TeV

    extraction = SpectrumExtraction(bkg_estimate=bkg_estimate(),
                                    obs_list=obs_list(),
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
                                                  edisp=0.2595896944765074,
                                                  containment=1,
                                                  )),
        (dict(containment_correction=True), dict(n_on=172,
                                                 sigma=24.98,
                                                 aeff=549861.8 * u.m ** 2,
                                                 edisp=0.2595896944765074,
                                                 containment=0.8525390663792395,
                                                 ))
    ])
    def test_extract(self, pars, results, obs_list, bkg_estimate):
        """Test quantitative output for various configs"""
        extraction = SpectrumExtraction(obs_list=obs_list,
                                        bkg_estimate=bkg_estimate,
                                        **pars)

        extraction.run()
        obs = extraction.observations[0]
        aeff_actual = obs.aeff.data.evaluate(energy=5 * u.TeV)
        edisp_actual = obs.edisp.data.evaluate(e_true=5 * u.TeV,
                                               e_reco=5.2 * u.TeV)

        assert_quantity_allclose(aeff_actual, results['aeff'], rtol=1e-3)
        assert_quantity_allclose(edisp_actual, results['edisp'], rtol=1e-3)

        # TODO: Introduce assert_stats_allclose
        n_on_actual = obs.total_stats.n_on
        sigma_actual = obs.total_stats.sigma
        containment_actual = obs.on_vector.areascal[60]

        assert n_on_actual == results['n_on']
        assert_allclose(sigma_actual, results['sigma'], atol=1e-2)
        assert_allclose(containment_actual, results['containment'], rtol=1e-3)

    def test_alpha(self, obs_list, bkg_estimate):
        bkg_estimate[0].a_off = 0
        bkg_estimate[1].a_off = 2
        extraction = SpectrumExtraction(obs_list=obs_list,
                                        bkg_estimate=bkg_estimate,
                                        max_alpha=0.2)
        extraction.run()
        assert len(extraction.observations) == 0

    def test_run(self, tmpdir, extraction):
        """Test the run method and check if files are written correctly"""
        extraction.run(outdir=tmpdir)
        testobs = SpectrumObservation.read(tmpdir / 'ogip_data' / 'pha_obs23523.fits')
        assert_quantity_allclose(testobs.aeff.data.data,
                                 extraction.observations[0].aeff.data.data)
        assert_quantity_allclose(testobs.on_vector.data.data,
                                 extraction.observations[0].on_vector.data.data)
        assert_quantity_allclose(testobs.on_vector.energy.nodes,
                                 extraction.observations[0].on_vector.energy.nodes)

    @requires_dependency('sherpa')
    def test_sherpa(self, tmpdir, extraction):
        """Same as above for files to be used with sherpa"""
        extraction.run(outdir=tmpdir, use_sherpa=True)

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

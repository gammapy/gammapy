# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest, assert_quantity_allclose
import astropy.units as u
import numpy as np
from astropy.utils.compat import NUMPY_LT_1_9
from numpy.testing import assert_allclose
from ...datasets import gammapy_extra
from ...spectrum import (
    PHACountsSpectrum,
    SpectrumObservationList,
    SpectrumObservation,
    SpectrumFit,
    SpectrumFitResult,
    models,
)
from ...utils.testing import (
    requires_dependency,
    requires_data,
    SHERPA_LT_4_8,
)
from ...utils.random import get_random_state

@pytest.mark.skipif('SHERPA_LT_4_8')
@requires_dependency('sherpa')
class TestFit:
    """Test fitter on counts spectra without any IRFs"""

    def setup(self):
        self.nbins = 30
        binning = np.logspace(-1, 1, self.nbins + 1) * u.TeV
        self.source_model = models.PowerLaw(index=2 * u.Unit(''),
                                            amplitude=1e5 / u.TeV,
                                            reference=0.1 * u.TeV)
        self.bkg_model = models.PowerLaw(index=3 * u.Unit(''),
                                         amplitude=1e4 / u.TeV,
                                         reference=0.1 * u.TeV)

        self.alpha = 0.1
        random_state = get_random_state(23)
        npred = self.source_model.integral(binning[:-1], binning[1:])
        source_counts = random_state.poisson(npred)
        self.src = PHACountsSpectrum(energy=binning, data=source_counts,
                                     backscal=1)

        npred_bkg = self.bkg_model.integral(binning[:-1], binning[1:])
        bkg_counts = random_state.poisson(npred_bkg)
        off_counts = random_state.poisson(npred_bkg * 1. / self.alpha)
        self.bkg = PHACountsSpectrum(energy=binning, data=bkg_counts)
        self.off = PHACountsSpectrum(energy=binning, data=off_counts,
                                     backscal=1. / self.alpha)

    def test_cash(self):
        """Simple CASH fit to the on vector"""
        obs = SpectrumObservation(on_vector=self.src)
        obs_list = SpectrumObservationList([obs])

        fit = SpectrumFit(obs_list=obs_list, model=self.source_model,
                          stat='cash', forward_folded=False)
        assert 'Spectrum' in str(fit)

        fit.predict_counts()
        assert_allclose(fit.predicted_counts[0][5], 660.5171280778071)

        fit.calc_statval()
        assert_allclose(np.sum(fit.statval[0]), -107346.5291329714)

        self.source_model.parameters.index = 1.12 * u.Unit('')
        fit.fit()
        # These values are check with sherpa fits, do not change
        assert_allclose(fit.model.parameters.index,
                        1.9955563477414806)
        assert_allclose(fit.model.parameters.amplitude.value,
                        100250.33102108649)

    def test_wstat(self):
        """WStat with on source and background spectrum"""
        on_vector = self.src.copy()
        on_vector.data.data += self.bkg.data.data
        obs = SpectrumObservation(on_vector=on_vector, off_vector=self.off)
        obs_list = SpectrumObservationList([obs])

        self.source_model.parameters.index = 1.12 * u.Unit('')
        fit = SpectrumFit(obs_list=obs_list, model=self.source_model,
                          stat='wstat', forward_folded=False)
        fit.fit()
        assert_allclose(fit.model.parameters.index,
                        1.997344538577775)
        assert_allclose(fit.model.parameters.amplitude.value,
                        100244.89943081759)
        assert_allclose(fit.result[0].statval, 30.022315611837342)

    def test_fit_range(self):
        """Test fit range without complication of thresholds"""
        obs = SpectrumObservation(on_vector=self.src)
        obs_list = SpectrumObservationList([obs])

        fit = SpectrumFit(obs_list=obs_list, model=self.source_model)
        assert np.sum(fit._bins_in_fit_range[0]) == self.nbins
        assert_allclose(fit.true_fit_range[0][-1], 10 * u.TeV)
        assert_allclose(fit.true_fit_range[0][0], 100 * u.GeV)

        fit.fit_range = [200, 600] * u.GeV
        assert np.sum(fit._bins_in_fit_range[0]) == 6
        assert_allclose(fit.true_fit_range[0][0], 0.21544347 * u.TeV)
        assert_allclose(fit.true_fit_range[0][-1], 541.1695265464 * u.GeV)


@pytest.mark.skipif('NUMPY_LT_1_9')
@pytest.mark.skipif('SHERPA_LT_4_8')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
class TestSpectralFit:
    """Test fitter in astrophysical scenario"""

    def setup(self):
        self.obs_list = SpectrumObservationList.read(
            '$GAMMAPY_EXTRA/datasets/hess-crab4_pha')

        self.pwl = models.PowerLaw(index=2 * u.Unit(''),
                                   amplitude=10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
                                   reference=1 * u.TeV)

        self.ecpl = models.ExponentialCutoffPowerLaw(
            index=2 * u.Unit(''),
            amplitude=10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV
        )

        # Example fit for one observation
        self.fit = SpectrumFit(self.obs_list[0:1], self.pwl)

    def test_basic_results(self):
        self.fit.fit()
        result = self.fit.result[0]
        assert self.fit.method == 'sherpa'
        assert_allclose(result.statval, 34.19706702533566)
        pars = result.model.parameters
        assert_quantity_allclose(pars.index,
                                 2.23957544167327)
        assert_quantity_allclose(pars.amplitude,
                                 2.018513315748709e-07 * u.Unit('m-2 s-1 TeV-1'))
        assert_allclose(result.npred[60], 0.5888275206035011)

    def test_basic_errors(self):
        self.fit.fit()
        self.fit.est_errors()
        result = self.fit.result[0]
        par_errors = result.model_with_uncertainties.parameters
        assert_allclose(par_errors.index.s, 0.09558428890966723)
        assert_allclose(par_errors.amplitude.s, 2.2154024177186417e-12)

    def test_npred(self):
        self.fit.fit()
        actual = self.fit.obs_list[0].predicted_counts(
            self.fit.result[0].model).data.data.value
        desired = self.fit.result[0].npred
        assert_allclose(actual, desired)

    @pytest.mark.xfail(reason='add stat per bin to fitresult in fit')
    def test_stats(self):
        stats = self.result.stats_per_bin()
        actual = np.sum(stats)
        desired = self.result.statval
        assert_allclose(actual, desired)

    def test_fit_range(self):
        # Fit range not restriced fit range should be the thresholds
        obs = self.fit.obs_list[0]
        desired = obs.on_vector.lo_threshold
        actual = self.fit.true_fit_range[0][0]
        assert_quantity_allclose(actual, desired)

        # Restrict fit range
        fit_range = [4, 20] * u.TeV
        self.fit.fit_range = fit_range

        range_bin = obs.on_vector.energy.find_node(fit_range[1])
        desired = obs.on_vector.energy.data[range_bin]
        actual = self.fit.true_fit_range[0][1]
        assert_quantity_allclose(actual, desired)

        # Make sure fit range is not extended below threshold
        fit_range = [0.001, 10] * u.TeV
        self.fit.fit_range = fit_range
        desired = obs.on_vector.lo_threshold
        actual = self.fit.true_fit_range[0][0]
        assert_quantity_allclose(actual, desired)

    @pytest.mark.xfail(reason='only simplex supported at the moment')
    def test_fit_method(self):
        self.fit.method_fit = "levmar"
        assert self.fit.method_fit.name == "levmar"
        self.fit.fit()
        result = self.fit.result[0]
        assert_quantity_allclose(result.model.parameters.index,
                                 2.2395184727047788)

    def test_ecpl_fit(self):
        fit = SpectrumFit(self.obs_list[0:1], self.ecpl)
        fit.fit()
        assert_quantity_allclose(fit.result[0].model.parameters.lambda_,
                                 0.0286068410937353 / u.TeV)

    def test_joint_fit(self):
        fit = SpectrumFit(self.obs_list, self.pwl)
        fit.fit()
        assert_quantity_allclose(fit.model.parameters.index,
                                 2.207512847977245)
        assert_quantity_allclose(fit.model.parameters.amplitude,
                                 2.3755942722352085e-07 * u.Unit('m-2 s-1 TeV-1'))

    def test_stacked_fit(self):
        stacked_obs = self.obs_list.stack()
        obs_list = SpectrumObservationList([stacked_obs])
        fit = SpectrumFit(obs_list, self.pwl)
        fit.fit()
        pars = fit.model.parameters
        assert_quantity_allclose(pars.index, 2.2462501437579476)
        assert_quantity_allclose(pars.amplitude,
                                 2.5160334568171844e-11 * u.Unit('cm-2 s-1 TeV-1'))

    def test_run(self, tmpdir):
        fit = SpectrumFit(obs_list, self.pwl)
        fit.run(outdir = tmpdir)


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_sherpa_fit(tmpdir):
    # this is to make sure that the written PHA files work with sherpa
    pha1 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23592.fits")

    import sherpa.astro.ui as sau
    from sherpa.models import PowLaw1D
    sau.load_pha(pha1)
    sau.set_stat('wstat')
    model = PowLaw1D('powlaw1d.default')
    model.ref = 1e9
    model.ampl = 1
    model.gamma = 2
    sau.set_model(model * 1e-20)
    sau.fit()
    assert_allclose(model.pars[0].val, 2.0198, atol=1e-4)
    assert_allclose(model.pars[2].val, 2.3564, atol=1e-4)

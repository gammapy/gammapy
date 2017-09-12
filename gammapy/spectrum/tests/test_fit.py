# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import assert_quantity_allclose
import pytest
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose
from ...utils.testing import requires_dependency, requires_data
from ...utils.random import get_random_state
from ...irf import EffectiveAreaTable
from ...spectrum import (
    PHACountsSpectrum,
    SpectrumObservationList,
    SpectrumObservation,
    SpectrumFit,
    SpectrumFitResult,
    models,
)


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
        self.src = PHACountsSpectrum(energy_lo=binning[:-1],
                                     energy_hi=binning[1:],
                                     data=source_counts,
                                     backscal=1)

        npred_bkg = self.bkg_model.integral(binning[:-1], binning[1:])

        bkg_counts = random_state.poisson(npred_bkg)
        off_counts = random_state.poisson(npred_bkg * 1. / self.alpha)
        self.bkg = PHACountsSpectrum(energy_lo=binning[:-1],
                                     energy_hi=binning[1:],
                                     data=bkg_counts)
        self.off = PHACountsSpectrum(energy_lo=binning[:-1],
                                     energy_hi=binning[1:],
                                     data=off_counts,
                                     backscal=1. / self.alpha)

    def test_cash(self):
        """Simple CASH fit to the on vector"""
        obs = SpectrumObservation(on_vector=self.src)
        obs_list = SpectrumObservationList([obs])

        fit = SpectrumFit(obs_list=obs_list, model=self.source_model,
                          stat='cash', forward_folded=False)
        assert 'Spectrum' in str(fit)

        fit.predict_counts()
        assert_allclose(fit.predicted_counts[0][0][5], 660.5171280778071)

        fit.calc_statval()
        assert_allclose(np.sum(fit.statval[0]), -107346.5291329714)

        self.source_model.parameters['index'].value = 1.12
        fit.fit()
        # These values are check with sherpa fits, do not change
        assert_allclose(fit.model.parameters['index'].value,
                        1.9955563477414806)
        assert_allclose(fit.model.parameters['amplitude'].value,
                        100250.33102108649)

    def test_cash_with_bkg(self):
        """Cash fit taking into account background model"""
        on_vector = self.src.copy()
        on_vector.data.data += self.bkg.data.data
        obs = SpectrumObservation(on_vector=on_vector, off_vector=self.off)
        obs_list = SpectrumObservationList([obs])

        self.source_model.parameters['index'].value = 1
        self.bkg_model.parameters['index'].value = 1
        fit = SpectrumFit(obs_list=obs_list, model=self.source_model,
                          stat='cash', forward_folded=False,
                          background_model=self.bkg_model)
        assert 'Background' in str(fit)

        fit.fit()
        assert_allclose(fit.result[0].model.parameters['index'].value,
                        1.996272386763962)
        assert_allclose(fit.background_model.parameters['index'].value,
                        2.9926225268193418)

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
        assert_allclose(fit.model.parameters['index'].value,
                        1.997344538577775)
        assert_allclose(fit.model.parameters['amplitude'].value,
                        100244.89943081759)
        assert_allclose(fit.result[0].statval, 30.022315611837342)

    def test_fit_range(self):
        """Test fit range without complication of thresholds"""
        obs = SpectrumObservation(on_vector=self.src)
        obs_list = SpectrumObservationList([obs])

        fit = SpectrumFit(obs_list=obs_list, model=self.source_model,
                          stat=None, forward_folded=False)
        assert np.sum(fit._bins_in_fit_range[0]) == self.nbins
        assert_allclose(fit.true_fit_range[0][-1], 10 * u.TeV)
        assert_allclose(fit.true_fit_range[0][0], 100 * u.GeV)

        fit.fit_range = [200, 600] * u.GeV
        assert np.sum(fit._bins_in_fit_range[0]) == 6
        assert_allclose(fit.true_fit_range[0][0], 0.21544347 * u.TeV)
        assert_allclose(fit.true_fit_range[0][-1], 541.1695265464 * u.GeV)

    def test_likelihood_profile(self):
        obs = SpectrumObservation(on_vector=self.src)
        fit = SpectrumFit(obs_list=obs, stat='cash', model=self.source_model,
                          forward_folded=False)
        fit.fit()
        true_idx = fit.result[0].model.parameters['index'].value
        scan_idx = np.linspace(0.95 * true_idx, 1.05 * true_idx, 100)
        profile = fit.likelihood_1d(model=fit.result[0].model, parname='index',
                                    parvals=scan_idx)
        argmin = np.argmin(profile)
        actual = scan_idx[argmin]
        assert_allclose(actual, true_idx, rtol=0.01)

    @requires_dependency('matplotlib')
    def test_plot(self):
        obs = SpectrumObservation(on_vector=self.src)
        fit = SpectrumFit(obs_list=obs, stat='cash', model=self.source_model,
                          forward_folded=False)

        scan_idx = np.linspace(1, 3, 20)
        fit.plot_likelihood_1d(model=self.source_model, parname='index',
                               parvals=scan_idx)
        # TODO: add assert, see issue 294


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
class TestSpectralFit:
    """Test fitter in astrophysical scenario"""

    def setup(self):
        self.obs_list = SpectrumObservationList.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha')

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
        self.fit = SpectrumFit(self.obs_list[0], self.pwl)

    def test_basic_results(self):
        self.fit.fit()
        result = self.fit.result[0]
        assert self.fit.method == 'sherpa'
        assert_allclose(result.statval, 32.838716584005645)
        pars = result.model.parameters
        assert_quantity_allclose(pars['index'].value, 2.2542312883476465)
        assert_quantity_allclose(pars['amplitude'].quantity,
                                 2.0082864582748925e-7 * u.Unit('m-2 s-1 TeV-1'))
        assert_allclose(result.npred_src[60], 0.563822994375907)

        with pytest.raises(ValueError):
            self.fit.result[0].to_table()

    def test_basic_errors(self):
        self.fit.fit()
        self.fit.est_errors()
        result = self.fit.result[0]
        assert_allclose(result.model.parameters.error('index'), 0.09787747219456712)
        assert_allclose(result.model.parameters.error('amplitude'), 2.1992645712596426e-12)
        self.fit.result[0].to_table()

    def test_compound(self):
        self.fit.model = self.fit.model * 2
        self.fit.fit()
        result = self.fit.result[0]
        pars = result.model.parameters
        assert_quantity_allclose(pars['index'].value, 2.2542315426423283)
        # amplitude should come out roughly * 0.5
        assert_quantity_allclose(pars['amplitude'].quantity,
                                 1.0243449507421302e-7 * u.Unit('m-2 s-1 TeV-1'))

    def test_areascal(self):
        areascal = np.ones(self.fit.obs_list[0].e_reco.nbins)
        areascal *= 0.5
        self.fit.obs_list[0].on_vector.areascal = areascal
        self.fit.fit()
        pars = self.fit.result[0].model.parameters
        assert_quantity_allclose(pars['index'].value, 2.2542312883476465)
        assert_quantity_allclose(pars['amplitude'].quantity,
                                 4.0165729155447114e-7 * u.Unit('m-2 s-1 TeV-1'))

    def test_npred(self):
        self.fit.fit()
        actual = self.fit.obs_list[0].predicted_counts(
            self.fit.result[0].model).data.data.value
        desired = self.fit.result[0].npred_src
        assert_allclose(actual, desired)

    def test_stats(self):
        self.fit.fit()
        stats = self.fit.result[0].stat_per_bin
        actual = np.sum(stats)
        desired = self.fit.result[0].statval
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
        desired = obs.on_vector.energy.lo[range_bin]
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
        assert_quantity_allclose(result.model.parametersr['index'].value,
                                 2.2395184727047788)

    def test_no_edisp(self):
        obs = self.obs_list[0]
        # Bring aeff in RECO space
        data = obs.aeff.data.evaluate(energy=obs.on_vector.energy.nodes)
        obs.aeff = EffectiveAreaTable(data=data,
                                      energy_lo=obs.on_vector.energy.lo,
                                      energy_hi=obs.on_vector.energy.hi)
        obs.edisp = None
        fit = SpectrumFit(obs_list=obs, model=self.pwl)
        fit.fit()
        assert_quantity_allclose(fit.result[0].model.parameters['index'].value,
                                 2.2960518556630887, atol=0.02)

    def test_ecpl_fit(self):
        fit = SpectrumFit(self.obs_list[0], self.ecpl)
        fit.fit()
        actual = fit.result[0].model.parameters['lambda_'].quantity
        assert_quantity_allclose(actual, 0.0341911861834517 / u.TeV)

    def test_joint_fit(self):
        fit = SpectrumFit(self.obs_list, self.pwl)
        fit.fit()
        actual = fit.model.parameters['index'].quantity
        assert_quantity_allclose(actual, 2.212325780417152)

        actual = fit.model.parameters['amplitude'].quantity
        assert_quantity_allclose(actual, 2.3621921135787887e-11 * u.Unit('cm-2 s-1 TeV-1'))

    def test_stacked_fit(self):
        stacked_obs = self.obs_list.stack()
        obs_list = SpectrumObservationList([stacked_obs])
        fit = SpectrumFit(obs_list, self.pwl)
        fit.fit()
        pars = fit.model.parameters
        assert_quantity_allclose(pars['index'].value, 2.2132304579760893)
        assert_quantity_allclose(pars['amplitude'].quantity,
                                 2.3618290865168973e-11 * u.Unit('cm-2 s-1 TeV-1'))

    def test_run(self, tmpdir):
        fit = SpectrumFit(self.obs_list, self.pwl)
        fit.run(outdir=tmpdir)

        read_result = SpectrumFitResult.from_yaml(tmpdir / 'fit_result_PowerLaw.yaml')

        desired = fit.result[0].model.evaluate_error(1 * u.TeV)
        actual = read_result.model.evaluate_error(1 * u.TeV)
        assert_quantity_allclose(actual, desired)

    def test_sherpa_fit(self, tmpdir):
        # this is to make sure that the written PHA files work with sherpa
        import sherpa.astro.ui as sau
        from sherpa.models import PowLaw1D

        self.obs_list.write(tmpdir, use_sherpa=True)
        filename = tmpdir / 'pha_obs23523.fits'
        sau.load_pha(str(filename))
        sau.set_stat('wstat')
        model = PowLaw1D('powlaw1d.default')
        model.ref = 1e9
        model.ampl = 1
        model.gamma = 2
        sau.set_model(model * 1e-20)
        sau.fit()
        assert_allclose(model.pars[0].val, 2.0881699260935838)
        assert_allclose(model.pars[2].val, 1.6234222129479836)

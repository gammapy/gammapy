# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
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
        self.source_model = models.PowerLaw(index=2,
                                            amplitude=1e5 / u.TeV,
                                            reference=0.1 * u.TeV)
        self.bkg_model = models.PowerLaw(index=3,
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
        # Currently it's necessary to specify a lifetime
        self.src.livetime = 1 * u.s

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
        assert_allclose(fit.predicted_counts[0][5], 660.5171, rtol=1e-5)

        fit.calc_statval()
        assert_allclose(np.sum(fit.statval[0]), -107346.5291, rtol=1e-5)

        self.source_model.parameters['index'].value = 1.12
        fit.fit()
        # These values are check with sherpa fits, do not change
        pars = fit.result[0].model.parameters
        assert_allclose(pars['index'].value, 1.995525, rtol=1e-3)
        assert_allclose(pars['amplitude'].value, 100245.9, rtol=1e-3)

    def test_wstat(self):
        """WStat with on source and background spectrum"""
        on_vector = self.src.copy()
        on_vector.data.data += self.bkg.data.data
        obs = SpectrumObservation(on_vector=on_vector, off_vector=self.off)
        obs_list = SpectrumObservationList([obs])

        self.source_model.parameters.index = 1.12
        fit = SpectrumFit(obs_list=obs_list, model=self.source_model,
                          stat='wstat', forward_folded=False)
        fit.fit()
        pars = fit.result[0].model.parameters
        assert_allclose(pars['index'].value, 1.997342, rtol=1e-3)
        assert_allclose(pars['amplitude'].value, 100245.187067, rtol=1e-3)
        assert_allclose(fit.result[0].statval, 30.022316, rtol=1e-3)

    def test_joint(self):
        """Test joint fit for obs with different energy binning"""
        obs1 = SpectrumObservation(on_vector=self.src)
        src_rebinned = self.src.rebin(2)
        obs2 = SpectrumObservation(on_vector=src_rebinned)
        fit = SpectrumFit(obs_list=[obs1, obs2], stat='cash',
                          model=self.source_model, forward_folded=False)
        fit.fit()
        pars = fit.result[0].model.parameters
        assert_allclose(pars['index'].value, 1.996456, rtol=1e-3)

    def test_fit_range(self):
        """Test fit range without complication of thresholds"""
        obs = SpectrumObservation(on_vector=self.src)
        obs_list = SpectrumObservationList([obs])

        fit = SpectrumFit(obs_list=obs_list, model=self.source_model,
                          stat=None, forward_folded=False)
        assert np.sum(fit._bins_in_fit_range[0]) == self.nbins
        assert_allclose(fit.true_fit_range[0][-1].value, 10)
        assert_allclose(fit.true_fit_range[0][0].value, 0.1)

        fit.fit_range = [200, 600] * u.GeV
        assert np.sum(fit._bins_in_fit_range[0]) == 6
        assert_allclose(fit.true_fit_range[0][0].value, 0.21544347, rtol=1e-5)
        assert_allclose(fit.true_fit_range[0][-1].value, 0.54117, rtol=1e-5)

        fit.fit_range = [0.11659144 + 1.e-5, 1. - 1.e-5] * u.TeV
        assert np.sum(fit._bins_in_fit_range[0]) == 14

        # Check different fit ranges for different observations
        on_vector2 = self.src.copy()
        obs2 = SpectrumObservation(on_vector=on_vector2)
        obs2.lo_threshold = 5 * u.TeV
        obs_list.append(obs2)
        fit = SpectrumFit(obs_list=obs_list, model=self.source_model,
                          stat=None, forward_folded=False)
        fit.fit_range = [2, 8] * u.TeV
        assert_allclose(fit.true_fit_range[0][0].value, 2.15443, rtol=1e-3)
        assert_allclose(fit.true_fit_range[1][0].value, 5.41169, rtol=1e-3)

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
@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectralFit:
    """Test fitter in astrophysical scenario"""

    def setup(self):
        self.obs_list = SpectrumObservationList.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha')

        self.pwl = models.PowerLaw(
            index=2,
            amplitude=10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
            reference=1 * u.TeV,
        )

        self.ecpl = models.ExponentialCutoffPowerLaw(
            index=2,
            amplitude=10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV
        )

        # Example fit for one observation
        self.fit = SpectrumFit(self.obs_list[0], self.pwl)

    @requires_dependency('iminuit')
    def test_basic_results(self):
        self.fit.method = 'iminuit'
        self.fit.fit()
        result = self.fit.result[0]
        assert_allclose(result.statval, 32.8387, rtol=1e-4)
        pars = result.model.parameters
        assert_allclose(pars['index'].value, 2.25423, rtol=1e-2)
        assert pars['amplitude'].unit == u.Unit('cm-2 s-1 TeV-1')
        assert_allclose(pars['amplitude'].value, 2.008654e-11, rtol=1e-2)
        assert_allclose(result.npred_src[60], 0.5638139, rtol=1e-3)
        self.fit.result[0].to_table()

    def test_basic_errors(self):
        self.fit.fit()
        self.fit.est_errors()
        result = self.fit.result[0]
        assert_allclose(result.model.parameters.error('index'), 0.097866953, rtol=1e-3)
        assert_allclose(result.model.parameters.error('amplitude'), 2.1994e-12, rtol=1e-3)
        self.fit.result[0].to_table()

    def test_compound(self):
        fit = SpectrumFit(self.obs_list[0], self.pwl * 2)
        fit.fit()
        result = fit.result[0]
        pars = result.model.parameters
        assert_allclose(pars['index'].value, 2.254163, rtol=1e-3)
        # amplitude should come out roughly * 0.5
        assert pars['amplitude'].unit == u.Unit('cm-2 s-1 TeV-1')
        assert_allclose(pars['amplitude'].value, 1.030963e-11, rtol=1e-3)

    def test_npred(self):
        self.fit.fit()
        actual = self.fit.obs_list[0].predicted_counts(self.fit.result[0].model).data.data.value
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
        assert actual.unit == 'keV'
        assert_allclose(actual.value, desired.value)

        # Restrict fit range
        fit_range = [4, 20] * u.TeV
        self.fit.fit_range = fit_range

        range_bin = obs.on_vector.energy.find_node(fit_range[1])
        desired = obs.on_vector.energy.lo[range_bin]
        actual = self.fit.true_fit_range[0][1]
        assert_allclose(actual.value, desired.value)

        # Make sure fit range is not extended below threshold
        fit_range = [0.001, 10] * u.TeV
        self.fit.fit_range = fit_range
        desired = obs.on_vector.lo_threshold
        actual = self.fit.true_fit_range[0][0]
        assert_allclose(actual.value, desired.value)

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
        assert_allclose(fit.result[0].model.parameters['index'].value, 2.296, atol=0.02)

    def test_ecpl_fit(self):
        self.ecpl.parameters.set_parameter_errors({
            'amplitude': 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
            'lambda': 0.1 / u.TeV
        })
        fit = SpectrumFit(self.obs_list[0], self.ecpl)
        fit.fit()
        actual = fit.result[0].model.parameters['lambda_'].quantity
        assert actual.unit == 'TeV-1'
        assert_allclose(actual.value, 0.034241, rtol=1e-3)

    def test_joint_fit(self):
        self.pwl.parameters.set_parameter_errors({
            'amplitude': 1e-11 * u.Unit('cm-2 s-1 TeV-1')
        })
        fit = SpectrumFit(self.obs_list, self.pwl)
        fit.fit()
        actual = fit.result[0].model.parameters['index'].value
        assert_allclose(actual, 2.21225, rtol=1e-3)

        actual = fit.result[0].model.parameters['amplitude'].quantity
        assert actual.unit == 'cm-2 s-1 TeV-1'
        assert_allclose(actual.value, 2.361871e-11, rtol=1e-3)

    def test_stacked_fit(self):
        stacked_obs = self.obs_list.stack()
        obs_list = SpectrumObservationList([stacked_obs])
        fit = SpectrumFit(obs_list, self.pwl)
        fit.fit()
        pars = fit.result[0].model.parameters
        assert_allclose(pars['index'].value, 2.21338, rtol=1e-3)
        assert u.Unit(pars['amplitude'].unit) == 'cm-2 s-1 TeV-1'
        assert_allclose(pars['amplitude'].value, 2.361827e-11, rtol=1e-3)

    def test_run(self, tmpdir):
        self.pwl.parameters.set_parameter_errors({
            'amplitude': 1e-11 * u.Unit('cm-2 s-1 TeV-1')
        })
        fit = SpectrumFit(self.obs_list, self.pwl)
        fit.run(outdir=tmpdir)

        read_result = SpectrumFitResult.from_yaml(tmpdir / 'fit_result_PowerLaw.yaml')

        desired = fit.result[0].model.evaluate_error(1 * u.TeV)
        actual = read_result.model.evaluate_error(1 * u.TeV)
        assert actual.unit == desired.unit
        assert_allclose(actual.value, desired.value)

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
        assert_allclose(model.pars[0].val, 2.0881699, rtol=1e-3)
        assert_allclose(model.pars[2].val, 1.6234222, rtol=1e-3)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose
from ...utils.testing import requires_data, requires_dependency
from ...utils.random import get_random_state
from ...utils.fitting import Fit
from ...irf import EffectiveAreaTable
from ...spectrum import (
    CountsSpectrum,
    models,
    SpectrumDatasetOnOff,
    SpectrumDataset,
    SpectrumDatasetOnOffStacker,
)


@requires_dependency("sherpa")
class TestFit:
    """Test fit on counts spectra without any IRFs"""

    def setup(self):
        self.nbins = 30
        binning = np.logspace(-1, 1, self.nbins + 1) * u.TeV
        self.source_model = models.PowerLaw(
            index=2, amplitude=1e5 / u.TeV, reference=0.1 * u.TeV
        )
        self.bkg_model = models.PowerLaw(
            index=3, amplitude=1e4 / u.TeV, reference=0.1 * u.TeV
        )

        self.alpha = 0.1
        random_state = get_random_state(23)
        npred = self.source_model.integral(binning[:-1], binning[1:])
        source_counts = random_state.poisson(npred)
        self.src = CountsSpectrum(
            energy_lo=binning[:-1], energy_hi=binning[1:], data=source_counts
        )
        # Currently it's necessary to specify a lifetime
        self.src.livetime = 1 * u.s

        npred_bkg = self.bkg_model.integral(binning[:-1], binning[1:])

        bkg_counts = random_state.poisson(npred_bkg)
        off_counts = random_state.poisson(npred_bkg * 1.0 / self.alpha)
        self.bkg = CountsSpectrum(
            energy_lo=binning[:-1], energy_hi=binning[1:], data=bkg_counts
        )
        self.off = CountsSpectrum(
            energy_lo=binning[:-1], energy_hi=binning[1:], data=off_counts
        )

    def test_cash(self):
        """Simple CASH fit to the on vector"""
        dataset = SpectrumDataset(model=self.source_model, counts=self.src)

        npred = dataset.npred().data
        assert_allclose(npred[5], 660.5171, rtol=1e-5)

        stat_val = dataset.likelihood()
        assert_allclose(stat_val, -107346.5291, rtol=1e-5)

        self.source_model.parameters["index"].value = 1.12

        fit = Fit([dataset])
        result = fit.run()

        # These values are check with sherpa fits, do not change
        pars = result.parameters
        assert_allclose(pars["index"].value, 1.995525, rtol=1e-3)
        assert_allclose(pars["amplitude"].value, 100245.9, rtol=1e-3)

    def test_fit_range(self):
        """Test fit range without complication of thresholds"""
        dataset = SpectrumDatasetOnOff(
            counts=self.src, mask_safe=np.ones(self.src.energy.nbin, dtype=bool)
        )
        dataset.model = self.source_model

        assert np.sum(dataset.mask_safe) == self.nbins
        e_min, e_max = dataset.energy_range

        assert_allclose(e_max.value, 10)
        assert_allclose(e_min.value, 0.1)

    def test_likelihood_profile(self):
        dataset = SpectrumDataset(
            model=self.source_model,
            counts=self.src,
            mask_safe=np.ones(self.src.energy.nbin, dtype=bool),
        )
        fit = Fit([dataset])
        result = fit.run()
        true_idx = result.parameters["index"].value
        values = np.linspace(0.95 * true_idx, 1.05 * true_idx, 100)
        profile = fit.likelihood_profile("index", values=values)
        actual = values[np.argmin(profile["likelihood"])]
        assert_allclose(actual, true_idx, rtol=0.01)


@requires_dependency("iminuit")
@requires_data()
class TestSpectralFit:
    """Test fit in astrophysical scenario"""

    def setup(self):
        path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
        obs1 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23523.fits")
        obs2 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23592.fits")
        self.obs_list = [obs1, obs2]

        self.pwl = models.PowerLaw(
            index=2, amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )

        self.ecpl = models.ExponentialCutoffPowerLaw(
            index=2,
            amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV,
        )

    def test_stats(self):
        dataset = self.obs_list[0]
        dataset.model = self.pwl

        fit = Fit([dataset])
        result = fit.run()

        stats = dataset.likelihood_per_bin()
        actual = np.sum(stats[dataset.mask_safe])

        desired = result.total_stat
        assert_allclose(actual, desired)

    def test_fit_range(self):
        # Fit range not restriced fit range should be the thresholds
        obs = self.obs_list[0]
        actual = obs.energy_range[0]

        assert actual.unit == "keV"
        assert_allclose(actual.value, 8.912509e08)

    def test_no_edisp(self):
        dataset = self.obs_list[0]

        # Bring aeff in RECO space
        energy = dataset.counts.energy.center
        data = dataset.aeff.data.evaluate(energy=energy)
        e_edges = dataset.counts.energy.edges

        dataset.aeff = EffectiveAreaTable(
            data=data, energy_lo=e_edges[:-1], energy_hi=e_edges[1:]
        )
        dataset.edisp = None
        dataset.model = self.pwl

        fit = Fit([dataset])
        result = fit.run()
        assert_allclose(result.parameters["index"].value, 2.7961, atol=0.02)

    def test_stacked_fit(self):
        obs_stacker = SpectrumDatasetOnOffStacker(self.obs_list)
        obs_stacker.run()

        dataset = obs_stacker.stacked_obs
        dataset.model = self.pwl

        fit = Fit([dataset])
        result = fit.run()
        pars = result.parameters

        assert_allclose(pars["index"].value, 2.7767, rtol=1e-3)
        assert u.Unit(pars["amplitude"].unit) == "cm-2 s-1 TeV-1"
        assert_allclose(pars["amplitude"].value, 5.191e-11, rtol=1e-3)

    @requires_dependency("sherpa")
    def test_sherpa_fit(self, tmpdir):
        # this is to make sure that the written PHA files work with sherpa
        import sherpa.astro.ui as sau
        from sherpa.models import PowLaw1D

        # TODO: this works a little bit, but some info and warnings
        # from Sherpa remain. Not sure what to do, OK as-is for now.
        import logging

        logging.getLogger("sherpa").setLevel("ERROR")

        for obs in self.obs_list:
            obs.to_ogip_files(str(tmpdir), use_sherpa=True)

        filename = tmpdir / "pha_obs23523.fits"
        sau.load_pha(str(filename))
        sau.set_stat("wstat")
        model = PowLaw1D("powlaw1d.default")
        model.ref = 1e9
        model.ampl = 1
        model.gamma = 2
        sau.set_model(model * 1e-20)
        sau.fit()
        assert_allclose(model.pars[0].val, 2.732, rtol=1e-3)
        assert_allclose(model.pars[2].val, 4.647, rtol=1e-3)

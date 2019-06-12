# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
import numpy as np
from ...utils.testing import requires_data, requires_dependency, mpl_plot_check
from ...utils.random import get_random_state
from ...irf import EffectiveAreaTable, EnergyDispersion
from ...utils.fitting import Fit
from ..models import PowerLaw, ConstantModel, ExponentialCutoffPowerLaw
from ...spectrum import (
    PHACountsSpectrum,
    SpectrumDatasetOnOff,
    SpectrumDataset,
    CountsSpectrum,
    SpectrumDatasetOnOffStacker,
)


@requires_dependency("iminuit")
class TestSpectrumDataset:
    """Test fit on counts spectra without any IRFs"""

    def setup(self):
        self.nbins = 30
        binning = np.logspace(-1, 1, self.nbins + 1) * u.TeV

        self.source_model = PowerLaw(
            index=2.1, amplitude=1e5 / u.TeV / u.s, reference=0.1 * u.TeV
        )

        self.livetime = 100 * u.s

        bkg_rate = np.ones(self.nbins) / u.s
        bkg_expected = bkg_rate * self.livetime

        self.bkg = CountsSpectrum(
            energy_lo=binning[:-1], energy_hi=binning[1:], data=bkg_expected
        )

        random_state = get_random_state(23)
        self.npred = (
            self.source_model.integral(binning[:-1], binning[1:]) * self.livetime
        )
        self.npred += bkg_expected
        source_counts = random_state.poisson(self.npred)

        self.src = CountsSpectrum(
            energy_lo=binning[:-1], energy_hi=binning[1:], data=source_counts
        )
        self.dataset = SpectrumDataset(
            self.source_model, self.src, self.livetime, None, None, None, self.bkg
        )

    def test_data_shape(self):
        assert self.dataset.data_shape[0] == self.nbins

    def test_energy_range(self):
        energy_range = self.dataset.energy_range
        assert energy_range.unit == u.TeV
        assert_allclose(energy_range.to_value('TeV'), [0.1, 10.])

    def test_cash(self):
        """Simple CASH fit to the on vector"""

        fit = Fit(self.dataset)
        result = fit.run()

        assert result.success
        assert "minuit" in repr(result)

        npred = self.dataset.npred().data.data.sum()
        assert_allclose(npred, self.npred.sum(), rtol=1e-3)
        assert_allclose(result.total_stat, -18087404.624, rtol=1e-3)

        pars = result.parameters
        assert_allclose(pars["index"].value, 2.1, rtol=1e-2)
        assert_allclose(pars.error("index"), 0.00127, rtol=1e-2)

        assert_allclose(pars["amplitude"].value, 1e5, rtol=1e-3)
        assert_allclose(pars.error("amplitude"), 153.450, rtol=1e-2)

    def test_fake(self):
        """Test the fake dataset"""
        fake_spectrum = self.dataset.fake(314)

        assert isinstance(fake_spectrum, CountsSpectrum)
        assert_allclose(fake_spectrum.energy.edges, self.dataset.counts.energy.edges)
        assert fake_spectrum.data.data.sum() == 907331

    def test_incorrect_mask(self):
        mask_fit = np.ones(self.nbins, dtype=np.dtype('float'))
        with pytest.raises(ValueError):
            SpectrumDataset(
                self.source_model, self.src, self.livetime, mask_fit, None, None, self.bkg
            )

    def test_set_model(self):
        aeff = EffectiveAreaTable.from_parametrization(self.src.energy.edges, 'HESS')
        edisp = EnergyDispersion.from_diagonal_response(self.src.energy.edges, self.src.energy.edges)
        dataset = SpectrumDataset(
            None, self.src, self.livetime, None, aeff, edisp, self.bkg
        )
        with pytest.raises(AttributeError):
            dataset.parameters

        dataset.model = self.source_model
        assert dataset.parameters[0] == self.source_model.parameters[0]

class TestSpectrumDatasetOnOff:
    """ Test ON OFF SpectrumDataset"""

    def setup(self):

        etrue = np.logspace(-1, 1, 10) * u.TeV
        self.e_true = etrue
        ereco = np.logspace(-1, 1, 5) * u.TeV
        elo = ereco[:-1]
        ehi = ereco[1:]

        self.aeff = EffectiveAreaTable(etrue[:-1], etrue[1:], np.ones(9) * u.cm ** 2)
        self.edisp = EnergyDispersion.from_diagonal_response(etrue, ereco)

        data = np.ones(elo.shape)
        data[-1] = 0  # to test stats calculation with empty bins
        self.on_counts = PHACountsSpectrum(
            elo, ehi, data , backscal=np.ones(elo.shape)
        )
        self.off_counts = PHACountsSpectrum(
            elo, ehi, np.ones(elo.shape) * 10, backscal=np.ones(elo.shape) * 10
        )
        self.on_counts.obs_id = "test"
        self.off_counts.obs_id = "test"

        self.livetime = 1000 * u.s

        self.dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            edisp=self.edisp,
            livetime=self.livetime,
        )


    def test_init_no_model(self):
        with pytest.raises(AttributeError):
            self.dataset.npred()

        assert hasattr(self.dataset, "parameters") == False

    def test_alpha(self):
        assert self.dataset.alpha.shape == (4,)
        assert_allclose(self.dataset.alpha, 0.1)

    def test_data_shape(self):
        assert self.dataset.data_shape == self.on_counts.data.data.shape

    def test_mask_safe_setter(self):
        with pytest.raises(ValueError):
            self.dataset.mask_safe = np.ones(self.dataset.data_shape, dtype='float')

    def test_reset_thresholds(self):
        counts = self.on_counts.copy()
        quality = np.ones(self.dataset.data_shape)
        quality[1:-1] = 0
        counts.quality = quality
        dataset = SpectrumDatasetOnOff(
            counts=counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            edisp=self.edisp,
            livetime=self.livetime,
        )

        assert_allclose(dataset.lo_threshold.to_value('TeV'),counts.energy.edges[1].to_value('TeV'))
        dataset.reset_thresholds()
        assert_allclose(dataset.lo_threshold.to_value('TeV'),counts.energy.edges[0].to_value('TeV'))

    def test_npred_no_edisp(self):
        const = 1 / u.TeV / u.cm ** 2 / u.s
        model = ConstantModel(const)
        livetime = 1 * u.s
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            model=model,
            livetime=livetime,
        )

        energy = self.aeff.energy.edges * self.aeff.energy.unit
        expected = self.aeff.data.data[0] * (energy[-1] - energy[0]) * const * livetime

        assert_allclose(dataset.npred().data.data.sum(), expected.value)

    @requires_dependency("matplotlib")
    def test_peek(self):
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            livetime=self.livetime,
            edisp=self.edisp,
        )
        with mpl_plot_check():
            dataset.peek()

    @requires_dependency("matplotlib")
    def test_plot_fit(self):
        model = PowerLaw()
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            model=model,
            aeff=self.aeff,
            livetime=self.livetime,
            edisp=self.edisp,
        )
        with mpl_plot_check():
            dataset.plot_fit()

    def test_to_from_ogip_files(self, tmpdir):
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            edisp=self.edisp,
            livetime=self.livetime,
        )
        dataset.to_ogip_files(outdir=tmpdir, overwrite=True)
        filename = tmpdir / self.on_counts.phafile
        newdataset = SpectrumDatasetOnOff.from_ogip_files(filename)

        assert_allclose(self.on_counts.data.data, newdataset.counts.data.data)
        assert_allclose(self.off_counts.data.data, newdataset.counts_off.data.data)
        assert_allclose(self.edisp.pdf_matrix, newdataset.edisp.pdf_matrix)

    def test_to_from_ogip_files_no_edisp(self, tmpdir):
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            aeff=self.aeff,
            livetime=self.livetime,
        )
        dataset.to_ogip_files(outdir=tmpdir, overwrite=True)
        filename = tmpdir / self.on_counts.phafile
        newdataset = SpectrumDatasetOnOff.from_ogip_files(filename)

        assert_allclose(self.on_counts.data.data, newdataset.counts.data.data)
        assert newdataset.counts_off is None
        assert newdataset.edisp is None


    def test_total_stats(self):
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            edisp=self.edisp,
            livetime=self.livetime,
        )

        assert dataset.total_stats.n_on == 3
        assert dataset.total_stats.n_off == 40
        assert dataset.total_stats.excess == -1

    def test_energy_mask(self):
        mask = self.dataset.counts.energy_mask(emin=0.3*u.TeV, emax=6*u.TeV)
        desired = [False, True, True, False]
        assert_allclose(mask, desired)

        mask = self.dataset.counts.energy_mask(emax=6*u.TeV)
        desired = [True, True, True, False]
        assert_allclose(mask, desired)

        mask = self.dataset.counts.energy_mask(emin=1*u.TeV)
        desired = [False, False, True, True]
        assert_allclose(mask, desired)


@requires_dependency("iminuit")
class TestSimpleFit:
    """Test fit on counts spectra without any IRFs"""

    def setup(self):
        self.nbins = 30
        binning = np.logspace(-1, 1, self.nbins + 1) * u.TeV
        self.source_model = PowerLaw(
            index=2, amplitude=1e5 / u.TeV, reference=0.1 * u.TeV
        )
        self.bkg_model = PowerLaw(index=3, amplitude=1e4 / u.TeV, reference=0.1 * u.TeV)

        self.alpha = 0.1
        random_state = get_random_state(23)
        npred = self.source_model.integral(binning[:-1], binning[1:])
        source_counts = random_state.poisson(npred)
        self.src = PHACountsSpectrum(
            energy_lo=binning[:-1],
            energy_hi=binning[1:],
            data=source_counts,
            backscal=1,
        )
        # Currently it's necessary to specify a lifetime
        self.src.livetime = 1 * u.s

        npred_bkg = self.bkg_model.integral(binning[:-1], binning[1:])

        bkg_counts = random_state.poisson(npred_bkg)
        off_counts = random_state.poisson(npred_bkg * 1.0 / self.alpha)
        self.bkg = PHACountsSpectrum(
            energy_lo=binning[:-1], energy_hi=binning[1:], data=bkg_counts
        )
        self.off = PHACountsSpectrum(
            energy_lo=binning[:-1],
            energy_hi=binning[1:],
            data=off_counts,
            backscal=1.0 / self.alpha,
        )

    def test_wstat(self):
        """WStat with on source and background spectrum"""
        on_vector = self.src.copy()
        on_vector.data.data += self.bkg.data.data
        obs = SpectrumDatasetOnOff(counts=on_vector, counts_off=self.off)
        obs.model = self.source_model

        self.source_model.parameters.index = 1.12

        fit = Fit(obs)
        result = fit.run()
        pars = self.source_model.parameters

        assert_allclose(pars["index"].value, 1.997342, rtol=1e-3)
        assert_allclose(pars["amplitude"].value, 100245.187067, rtol=1e-3)
        assert_allclose(result.total_stat, 30.022316, rtol=1e-3)

    def test_joint(self):
        """Test joint fit for obs with different energy binning"""
        on_vector = self.src.copy()
        on_vector.data.data += self.bkg.data.data
        obs1 = SpectrumDatasetOnOff(counts=on_vector, counts_off=self.off)
        obs1.model = self.source_model

        src_rebinned = self.src.rebin(2)
        bkg_rebinned = self.off.rebin(2)
        src_rebinned.data.data += self.bkg.rebin(2).data.data

        obs2 = SpectrumDatasetOnOff(counts=src_rebinned, counts_off=bkg_rebinned)
        obs2.model = self.source_model

        fit = Fit([obs1, obs2])
        fit.run()
        pars = self.source_model.parameters
        assert_allclose(pars["index"].value, 1.920686, rtol=1e-3)


@requires_data()
@requires_dependency("iminuit")
class TestSpectralFit:
    """Test fit in astrophysical scenario"""

    def setup(self):
        path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
        obs1 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23523.fits")
        obs2 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23592.fits")
        self.obs_list = [obs1, obs2]

        self.pwl = PowerLaw(
            index=2, amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )

        self.ecpl = ExponentialCutoffPowerLaw(
            index=2,
            amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV,
        )

        # Example fit for one observation
        self.obs_list[0].model = self.pwl
        self.fit = Fit(self.obs_list[0])

    def set_model(self, model):
        for obs in self.obs_list:
            obs.model = model

    @requires_dependency("iminuit")
    def test_basic_results(self):
        self.set_model(self.pwl)
        result = self.fit.run()
        pars = self.fit.datasets.parameters

        assert self.pwl is self.obs_list[0].model

        assert_allclose(result.total_stat, 38.343, rtol=1e-3)
        assert_allclose(pars["index"].value, 2.817, rtol=1e-3)
        assert pars["amplitude"].unit == "cm-2 s-1 TeV-1"
        assert_allclose(pars["amplitude"].value, 5.142e-11, rtol=1e-3)
        assert_allclose(self.obs_list[0].npred().data.data[60], 0.6102, rtol=1e-3)
        pars.to_table()

    def test_basic_errors(self):
        self.set_model(self.pwl)
        self.fit.run()
        pars = self.fit.datasets.parameters

        assert_allclose(pars.error("index"), 0.1496, rtol=1e-3)
        assert_allclose(pars.error("amplitude"), 6.423e-12, rtol=1e-3)
        pars.to_table()

    def test_compound(self):
        model = self.pwl * 2
        self.set_model(model)
        fit = Fit(self.obs_list[0])
        fit.run()
        pars = fit.datasets.parameters

        assert_allclose(pars["index"].value, 2.8166, rtol=1e-3)
        p = pars["amplitude"]
        assert p.unit == "cm-2 s-1 TeV-1"
        assert_allclose(p.value, 5.0714e-12, rtol=1e-3)

    def test_ecpl_fit(self):
        self.set_model(self.ecpl)
        fit = Fit(self.obs_list[0])
        fit.run()

        actual = fit.datasets.parameters["lambda_"].quantity
        assert actual.unit == "TeV-1"
        assert_allclose(actual.value, 0.145215, rtol=1e-2)

    def test_joint_fit(self):
        self.set_model(self.pwl)
        fit = Fit(self.obs_list)
        fit.run()
        actual = fit.datasets.parameters["index"].value
        assert_allclose(actual, 2.7806, rtol=1e-3)

        actual = fit.datasets.parameters["amplitude"].quantity
        assert actual.unit == "cm-2 s-1 TeV-1"
        assert_allclose(actual.value, 5.200e-11, rtol=1e-3)


def _read_hess_obs():
    path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
    obs1 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23523.fits")
    obs2 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23592.fits")
    return [obs1, obs2]


@requires_dependency("sherpa")
@requires_data("gammapy-data")
def make_observation_list():
    """obs with dummy IRF"""
    nbin = 3
    energy = np.logspace(-1, 1, nbin + 1) * u.TeV
    livetime = 2 * u.h
    data_on = np.arange(nbin)
    dataoff_1 = np.ones(3)
    dataoff_2 = np.ones(3) * 3
    dataoff_1[1] = 0
    dataoff_2[1] = 0
    on_vector = PHACountsSpectrum(
        energy_lo=energy[:-1], energy_hi=energy[1:], data=data_on, backscal=1
    )
    off_vector1 = PHACountsSpectrum(
        energy_lo=energy[:-1], energy_hi=energy[1:], data=dataoff_1, backscal=2
    )
    off_vector2 = PHACountsSpectrum(
        energy_lo=energy[:-1], energy_hi=energy[1:], data=dataoff_2, backscal=4
    )
    aeff = EffectiveAreaTable(
        energy_lo=energy[:-1], energy_hi=energy[1:], data=np.ones(nbin) * 1e5 * u.m ** 2
    )
    edisp = EnergyDispersion.from_gauss(e_true=energy, e_reco=energy, sigma=0.2, bias=0)
    on_vector.livetime = livetime
    on_vector.obs_id = 2
    obs1 = SpectrumDatasetOnOff(
        counts=on_vector,
        counts_off=off_vector1,
        aeff=aeff,
        edisp=edisp,
        livetime=livetime,
    )
    obs2 = SpectrumDatasetOnOff(
        counts=on_vector,
        counts_off=off_vector2,
        aeff=aeff,
        edisp=edisp,
        livetime=livetime,
    )

    obs_list = [obs1, obs2]
    return obs_list


@requires_data("gammapy-data")
class TestSpectrumDatasetOnOffStacker:
    def setup(self):
        self.obs_list = _read_hess_obs()

        # Change threshold to make stuff more interesting
        self.obs_list[0].mask_safe = self.obs_list[0].counts.energy_mask(emin=1.2 * u.TeV, emax=50 * u.TeV)

        self.obs_list[1].mask_safe &= self.obs_list[0].counts.energy_mask(emax=20 * u.TeV)

        self.obs_stacker = SpectrumDatasetOnOffStacker(self.obs_list)
        self.obs_stacker.run()

    def test_basic(self):
        assert "Stacker" in str(self.obs_stacker)
        assert "stacked" in str(self.obs_stacker.stacked_obs.counts.phafile)
        counts1 = self.obs_list[0].total_stats_safe_range.n_on
        counts2 = self.obs_list[1].total_stats_safe_range.n_on
        summed_counts = counts1 + counts2
        stacked_counts = self.obs_stacker.stacked_obs.total_stats.n_on
        assert summed_counts == stacked_counts

    def test_thresholds(self):

        e_min, e_max = self.obs_stacker.stacked_obs.energy_range

        assert e_min.unit == "keV"
        assert_allclose(e_min.value, 8.912509e08, rtol=1e-3)

        assert e_max.unit == "keV"
        assert_allclose(e_max.value, 4.466836e10, rtol=1e-3)

    def test_verify_npred(self):
        """Veryfing npred is preserved during the stacking"""
        pwl = PowerLaw(
            index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )
        self.obs_stacker.stacked_obs.model = pwl
        npred_stacked = self.obs_stacker.stacked_obs.npred().data.data
        npred_summed = np.zeros_like(npred_stacked)

        for obs in self.obs_list:
            obs.model = pwl
            npred_summed[obs.mask_safe] += obs.npred().data.data[obs.mask_safe]

        assert_allclose(npred_stacked, npred_summed)

    def test_stack_backscal(self):
        """Verify backscal stacking """
        obs_list = make_observation_list()
        obs_stacker = SpectrumDatasetOnOffStacker(obs_list)
        obs_stacker.run()
        assert_allclose(obs_stacker.stacked_obs.alpha[0], 1.25 / 4.0)
        # When the OFF stack observation counts=0, the alpha is averaged on the total OFF counts for each run.
        assert_allclose(obs_stacker.stacked_obs.alpha[1], 2.5 / 8.0)

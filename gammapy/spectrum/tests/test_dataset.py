# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.data import GTI
from gammapy.irf import EffectiveAreaTable, EnergyDispersion
from gammapy.modeling import Datasets, Fit
from gammapy.modeling.models import (
    ConstantSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
)
from gammapy.spectrum import CountsSpectrum, SpectrumDataset, SpectrumDatasetOnOff
from gammapy.utils.random import get_random_state
from gammapy.utils.testing import (
    assert_time_allclose,
    mpl_plot_check,
    requires_data,
    requires_dependency,
)
from gammapy.utils.time import time_ref_to_dict


@requires_dependency("iminuit")
class TestSpectrumDataset:
    """Test fit on counts spectra without any IRFs"""

    def setup(self):
        self.nbins = 30
        binning = np.logspace(-1, 1, self.nbins + 1) * u.TeV

        self.source_model = PowerLawSpectralModel(
            index=2.1, amplitude=1e5 * u.Unit("cm-2 s-1 TeV-1"), reference=0.1 * u.TeV
        )

        self.livetime = 100 * u.s
        aeff = EffectiveAreaTable.from_constant(binning, "1 cm2")

        bkg_rate = np.ones(self.nbins) / u.s
        bkg_expected = (bkg_rate * self.livetime).to_value("")

        self.bkg = CountsSpectrum(
            energy_lo=binning[:-1], energy_hi=binning[1:], data=bkg_expected
        )

        random_state = get_random_state(23)
        flux = self.source_model.integral(binning[:-1], binning[1:])
        self.npred = (flux * aeff.data.data[0] * self.livetime).to_value("")
        self.npred += bkg_expected
        source_counts = random_state.poisson(self.npred)

        self.src = CountsSpectrum(
            energy_lo=binning[:-1], energy_hi=binning[1:], data=source_counts
        )
        self.dataset = SpectrumDataset(
            model=self.source_model,
            counts=self.src,
            aeff=aeff,
            livetime=self.livetime,
            background=self.bkg,
        )

    def test_data_shape(self):
        assert self.dataset.data_shape[0] == self.nbins

    def test_energy_range(self):
        energy_range = self.dataset.energy_range
        assert energy_range.unit == u.TeV
        assert_allclose(energy_range.to_value("TeV"), [0.1, 10.0])

    def test_cash(self):
        """Simple CASH fit to the on vector"""
        fit = Fit(self.dataset)
        result = fit.run()

        assert result.success
        assert "minuit" in repr(result)

        npred = self.dataset.npred().data.sum()
        assert_allclose(npred, self.npred.sum(), rtol=1e-3)
        assert_allclose(result.total_stat, -18087404.624, rtol=1e-3)

        pars = result.parameters
        assert_allclose(pars["index"].value, 2.1, rtol=1e-2)
        assert_allclose(pars.error("index"), 0.00127, rtol=1e-2)

        assert_allclose(pars["amplitude"].value, 1e5, rtol=1e-3)
        assert_allclose(pars.error("amplitude"), 153.450, rtol=1e-2)

    def test_fake(self):
        """Test the fake dataset"""
        real_dataset = self.dataset.copy()
        self.dataset.fake(314)
        assert real_dataset.counts.data.shape == self.dataset.counts.data.shape
        assert real_dataset.background.data.sum() == self.dataset.background.data.sum()
        assert int(real_dataset.counts.data.sum()) == 907010
        assert self.dataset.counts.data.sum() == 907331

    def test_incorrect_mask(self):
        mask_fit = np.ones(self.nbins, dtype=np.dtype("float"))
        with pytest.raises(ValueError):
            SpectrumDataset(
                model=self.source_model,
                counts=self.src,
                livetime=self.livetime,
                mask_fit=mask_fit,
                background=self.bkg,
            )

    def test_set_model(self):
        aeff = EffectiveAreaTable.from_parametrization(self.src.energy.edges, "HESS")
        edisp = EnergyDispersion.from_diagonal_response(
            self.src.energy.edges, self.src.energy.edges
        )
        dataset = SpectrumDataset(
            None, self.src, self.livetime, None, aeff, edisp, self.bkg
        )
        with pytest.raises(AttributeError):
            dataset.parameters

        dataset.model = self.source_model
        assert dataset.parameters[0] == self.source_model.parameters[0]

    def test_str(self):
        assert "SpectrumDataset" in str(self.dataset)

    def test_spectrumdataset_create(self):
        e_reco = u.Quantity([0.1, 1, 10.0], "TeV")
        e_true = u.Quantity([0.05, 0.5, 5, 20.0], "TeV")
        empty_dataset = SpectrumDataset.create(e_reco, e_true)

        assert empty_dataset.counts.total_counts == 0
        assert empty_dataset.data_shape[0] == 2
        assert empty_dataset.background.total_counts == 0
        assert empty_dataset.background.energy.nbin == 2
        assert empty_dataset.aeff.data.axis("energy").nbin == 3
        assert empty_dataset.edisp.data.axis("e_reco").nbin == 2
        assert empty_dataset.livetime.value == 0
        assert len(empty_dataset.gti.table) == 0
        assert empty_dataset.energy_range[0] is None
        assert_allclose(empty_dataset.mask_safe, 0)

    def test_spectrum_dataset_stack_diagonal_safe_mask(self):
        aeff = EffectiveAreaTable.from_parametrization(self.src.energy.edges, "HESS")
        edisp = EnergyDispersion.from_diagonal_response(
            self.src.energy.edges, self.src.energy.edges
        )
        livetime = self.livetime
        dataset1 = SpectrumDataset(
            counts=self.src.copy(),
            livetime=livetime,
            aeff=aeff,
            edisp=edisp,
            background=self.bkg.copy(),
        )

        livetime2 = 0.5 * livetime
        aeff2 = EffectiveAreaTable(
            self.src.energy.edges[:-1], self.src.energy.edges[1:], 2 * aeff.data.data
        )
        bkg2 = CountsSpectrum(
            self.src.energy.edges[:-1],
            self.src.energy.edges[1:],
            data=2 * self.bkg.data,
        )
        safe_mask2 = np.ones_like(self.src.data, bool)
        safe_mask2[0] = False
        dataset2 = SpectrumDataset(
            counts=self.src.copy(),
            livetime=livetime2,
            aeff=aeff2,
            edisp=edisp,
            background=bkg2,
            mask_safe=safe_mask2,
        )
        dataset1.stack(dataset2)

        assert_allclose(dataset1.counts.data[1:], self.src.data[1:] * 2)
        assert_allclose(dataset1.counts.data[0], self.src.data[0])
        assert dataset1.livetime == 1.5 * self.livetime
        assert_allclose(dataset1.background.data[1:], 3 * self.bkg.data[1:])
        assert_allclose(dataset1.background.data[0], self.bkg.data[0])
        assert_allclose(
            dataset1.aeff.data.data.to_value("m2"),
            4.0 / 3 * aeff.data.data.to_value("m2"),
        )
        assert_allclose(dataset1.edisp.pdf_matrix[1:], edisp.pdf_matrix[1:])
        assert_allclose(dataset1.edisp.pdf_matrix[0], 0.5 * edisp.pdf_matrix[0])

    def test_spectrum_dataset_stack_nondiagonal_no_bkg(self):
        aeff = EffectiveAreaTable.from_parametrization(self.src.energy.edges, "HESS")
        edisp1 = EnergyDispersion.from_gauss(
            self.src.energy.edges, self.src.energy.edges, 0.1, 0.0
        )
        livetime = self.livetime
        dataset1 = SpectrumDataset(
            counts=None, livetime=livetime, aeff=aeff, edisp=edisp1, background=None
        )

        livetime2 = livetime
        aeff2 = EffectiveAreaTable(
            self.src.energy.edges[:-1], self.src.energy.edges[1:], aeff.data.data
        )
        edisp2 = EnergyDispersion.from_gauss(
            self.src.energy.edges, self.src.energy.edges, 0.2, 0.0
        )
        dataset2 = SpectrumDataset(
            counts=self.src.copy(),
            livetime=livetime2,
            aeff=aeff2,
            edisp=edisp2,
            background=None,
        )
        dataset1.stack(dataset2)

        assert dataset1.counts is None
        assert dataset1.background is None
        assert dataset1.livetime == 2 * self.livetime
        assert_allclose(
            dataset1.aeff.data.data.to_value("m2"), aeff.data.data.to_value("m2")
        )
        assert_allclose(dataset1.edisp.get_bias(1 * u.TeV), 0.0, atol=1e-3)
        assert_allclose(dataset1.edisp.get_resolution(1 * u.TeV), 0.1581, atol=1e-2)


class TestSpectrumOnOff:
    """ Test ON OFF SpectrumDataset"""

    def setup(self):
        etrue = np.logspace(-1, 1, 10) * u.TeV
        self.e_true = etrue
        ereco = np.logspace(-1, 1, 5) * u.TeV
        elo = ereco[:-1]
        ehi = ereco[1:]
        self.e_reco = ereco
        self.aeff = EffectiveAreaTable(etrue[:-1], etrue[1:], np.ones(9) * u.cm ** 2)
        self.edisp = EnergyDispersion.from_diagonal_response(etrue, ereco)

        data = np.ones(elo.shape)
        data[-1] = 0  # to test stats calculation with empty bins
        self.on_counts = CountsSpectrum(elo, ehi, data)
        self.off_counts = CountsSpectrum(elo, ehi, np.ones(elo.shape) * 10)

        start = u.Quantity([0], "s")
        stop = u.Quantity([1000], "s")
        time_ref = Time("2010-01-01 00:00:00.0")
        self.gti = GTI.create(start, stop, time_ref)
        self.livetime = self.gti.time_sum

        self.dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            edisp=self.edisp,
            livetime=self.livetime,
            acceptance=np.ones(elo.shape),
            acceptance_off=np.ones(elo.shape) * 10,
            name="test",
            gti=self.gti,
        )

    def test_spectrumdatasetonoff_create(self):
        e_reco = u.Quantity([0.1, 1, 10.0], "TeV")
        e_true = u.Quantity([0.05, 0.5, 5, 20.0], "TeV")
        empty_dataset = SpectrumDatasetOnOff.create(e_reco, e_true)

        assert empty_dataset.counts.total_counts == 0
        assert empty_dataset.data_shape[0] == 2
        assert empty_dataset.counts_off.total_counts == 0
        assert empty_dataset.counts_off.energy.nbin == 2
        assert_allclose(empty_dataset.acceptance_off, 1)
        assert_allclose(empty_dataset.acceptance, 1)
        assert empty_dataset.acceptance.shape[0] == 2
        assert empty_dataset.acceptance_off.shape[0] == 2
        assert empty_dataset.livetime.value == 0
        assert len(empty_dataset.gti.table) == 0
        assert empty_dataset.energy_range[0] is None

    def test_create_stack(self):
        stacked = SpectrumDatasetOnOff.create(self.e_reco, self.e_true)
        stacked.stack(self.dataset)
        assert_allclose(stacked.energy_range.value, self.dataset.energy_range.value)

    def test_init_no_model(self):
        with pytest.raises(AttributeError):
            self.dataset.npred()

        assert not hasattr(self.dataset, "parameters")

    def test_alpha(self):
        assert self.dataset.alpha.shape == (4,)
        assert_allclose(self.dataset.alpha, 0.1)

    def test_data_shape(self):
        assert self.dataset.data_shape == self.on_counts.data.shape

    def test_npred_no_edisp(self):
        const = 1 * u.Unit("cm-2 s-1 TeV-1")
        model = ConstantSpectralModel(const=const)
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

        assert_allclose(dataset.npred_sig().data.sum(), expected.value)

    @requires_dependency("matplotlib")
    def test_peek(self):
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            livetime=self.livetime,
            edisp=self.edisp,
            acceptance=1,
            acceptance_off=10,
        )
        with mpl_plot_check():
            dataset.peek()

    @requires_dependency("matplotlib")
    def test_plot_fit(self):
        model = PowerLawSpectralModel()
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            model=model,
            aeff=self.aeff,
            livetime=self.livetime,
            edisp=self.edisp,
            acceptance=1,
            acceptance_off=10,
        )
        with mpl_plot_check():
            dataset.plot_fit()

    def test_to_from_ogip_files(self, tmp_path):
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            edisp=self.edisp,
            livetime=self.livetime,
            mask_safe=np.ones(self.on_counts.energy.nbin, dtype=bool),
            acceptance=1,
            acceptance_off=10,
            name="test",
            gti=self.gti,
        )
        dataset.to_ogip_files(outdir=tmp_path)
        newdataset = SpectrumDatasetOnOff.from_ogip_files(tmp_path / "pha_obstest.fits")

        assert_allclose(self.on_counts.data, newdataset.counts.data)
        assert_allclose(self.off_counts.data, newdataset.counts_off.data)
        assert_allclose(self.edisp.pdf_matrix, newdataset.edisp.pdf_matrix)
        assert_time_allclose(newdataset.gti.time_start, dataset.gti.time_start)

    def test_to_from_ogip_files_no_edisp(self, tmp_path):
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            aeff=self.aeff,
            livetime=self.livetime,
            mask_safe=np.ones(self.on_counts.energy.nbin, dtype=bool),
            acceptance=1,
            name="test",
        )
        dataset.to_ogip_files(outdir=tmp_path)
        newdataset = SpectrumDatasetOnOff.from_ogip_files(tmp_path / "pha_obstest.fits")

        assert_allclose(self.on_counts.data, newdataset.counts.data)
        assert newdataset.counts_off is None
        assert newdataset.edisp is None
        assert newdataset.gti is None

    def test_energy_mask(self):
        mask = self.dataset.counts.energy_mask(emin=0.3 * u.TeV, emax=6 * u.TeV)
        desired = [False, True, True, False]
        assert_allclose(mask, desired)

        mask = self.dataset.counts.energy_mask(emax=6 * u.TeV)
        desired = [True, True, True, False]
        assert_allclose(mask, desired)

        mask = self.dataset.counts.energy_mask(emin=1 * u.TeV)
        desired = [False, False, True, True]
        assert_allclose(mask, desired)

    def test_str(self):
        model = PowerLawSpectralModel()
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            model=model,
            aeff=self.aeff,
            livetime=self.livetime,
            edisp=self.edisp,
            acceptance=1,
            acceptance_off=10,
        )
        assert "SpectrumDatasetOnOff" in str(dataset)
        assert "wstat" in str(dataset)

    def test_fake(self):
        """Test the fake dataset"""
        source_model = PowerLawSpectralModel()
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            model=source_model,
            aeff=self.aeff,
            livetime=self.livetime,
            edisp=self.edisp,
            acceptance=1,
            acceptance_off=10,
        )
        real_dataset = dataset.copy()
        # Define background model counts
        elo = self.on_counts.energy.edges[:-1]
        ehi = self.on_counts.energy.edges[1:]
        data = np.ones(self.on_counts.data.shape)
        background_model = CountsSpectrum(elo, ehi, data)
        dataset.fake(background_model=background_model, random_state=314)

        assert real_dataset.counts.data.shape == dataset.counts.data.shape
        assert real_dataset.counts_off.data.shape == dataset.counts_off.data.shape
        assert (
            real_dataset.counts.energy.center.mean()
            == dataset.counts.energy.center.mean()
        )
        assert real_dataset.acceptance.mean() == dataset.acceptance.mean()
        assert real_dataset.acceptance_off.mean() == dataset.acceptance_off.mean()
        assert dataset.counts_off.data.sum() == 39
        assert dataset.counts.data.sum() == 5

    def test_info_dict(self):
        info_dict = self.dataset.info_dict()

        assert_allclose(info_dict["n_on"], 3)
        assert_allclose(info_dict["n_off"], 40)
        assert_allclose(info_dict["a_on"], 1)
        assert_allclose(info_dict["a_off"], 10)

        assert_allclose(info_dict["alpha"], 0.1)
        assert_allclose(info_dict["excess"], -1)
        assert_allclose(info_dict["livetime"].value, 1e3)

        assert info_dict["name"] == "test"


@requires_data()
@requires_dependency("iminuit")
class TestSpectralFit:
    """Test fit in astrophysical scenario"""

    def setup(self):
        path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
        obs1 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23523.fits")
        obs2 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23592.fits")
        self.obs_list = [obs1, obs2]

        self.pwl = PowerLawSpectralModel(
            index=2, amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )

        self.ecpl = ExpCutoffPowerLawSpectralModel(
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
        assert_allclose(self.obs_list[0].npred().data[60], 0.6102, rtol=1e-3)
        pars.to_table()

    def test_basic_errors(self):
        self.set_model(self.pwl)
        result = self.fit.run()
        pars = result.parameters

        assert_allclose(pars.error("index"), 0.1496, rtol=1e-3)
        assert_allclose(pars.error("amplitude"), 6.423e-12, rtol=1e-3)
        pars.to_table()

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


def make_gti(times, time_ref="2010-01-01"):
    meta = time_ref_to_dict(time_ref)
    table = Table(times, meta=meta)
    return GTI(table)


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
    on_vector = CountsSpectrum(
        energy_lo=energy[:-1], energy_hi=energy[1:], data=data_on
    )
    off_vector1 = CountsSpectrum(
        energy_lo=energy[:-1], energy_hi=energy[1:], data=dataoff_1
    )
    off_vector2 = CountsSpectrum(
        energy_lo=energy[:-1], energy_hi=energy[1:], data=dataoff_2
    )
    aeff = EffectiveAreaTable.from_constant(energy, "1 cm2")
    edisp = EnergyDispersion.from_gauss(e_true=energy, e_reco=energy, sigma=0.2, bias=0)

    time_ref = Time("2010-01-01")
    gti1 = make_gti({"START": [5, 6, 1, 2], "STOP": [8, 7, 3, 4]}, time_ref=time_ref)
    gti2 = make_gti({"START": [14], "STOP": [15]}, time_ref=time_ref)

    obs1 = SpectrumDatasetOnOff(
        counts=on_vector,
        counts_off=off_vector1,
        aeff=aeff,
        edisp=edisp,
        livetime=livetime,
        mask_safe=np.ones(on_vector.energy.nbin, dtype=bool),
        acceptance=1,
        acceptance_off=2,
        name="2",
        gti=gti1,
    )
    obs2 = SpectrumDatasetOnOff(
        counts=on_vector,
        counts_off=off_vector2,
        aeff=aeff,
        edisp=edisp,
        livetime=livetime,
        mask_safe=np.ones(on_vector.energy.nbin, dtype=bool),
        acceptance=1,
        acceptance_off=4,
        name="2",
        gti=gti2,
    )

    obs_list = [obs1, obs2]
    return obs_list


@requires_data("gammapy-data")
class TestSpectrumDatasetOnOffStack:
    def setup(self):
        self.obs_list = _read_hess_obs()
        # Change threshold to make stuff more interesting
        self.obs_list[0].mask_safe = self.obs_list[0].counts.energy_mask(
            emin=1.2 * u.TeV, emax=50 * u.TeV
        )

        self.obs_list[1].mask_safe &= self.obs_list[0].counts.energy_mask(
            emax=20 * u.TeV
        )

        self.stacked_dataset = self.obs_list[0].copy()
        self.stacked_dataset.stack(self.obs_list[1])

    def test_basic(self):
        obs_1, obs_2 = self.obs_list

        counts1 = obs_1.counts.data[obs_1.mask_safe].sum()
        counts2 = obs_2.counts.data[obs_2.mask_safe].sum()
        summed_counts = counts1 + counts2

        stacked_counts = self.stacked_dataset.counts.data.sum()

        off1 = obs_1.counts_off.data[obs_1.mask_safe].sum()
        off2 = obs_2.counts_off.data[obs_2.mask_safe].sum()
        summed_off = off1 + off2
        stacked_off = self.stacked_dataset.counts_off.data.sum()

        assert summed_counts == stacked_counts
        assert summed_off == stacked_off

    def test_thresholds(self):
        e_min, e_max = self.stacked_dataset.energy_range

        assert e_min.unit == "keV"
        assert_allclose(e_min.value, 8.912509e08, rtol=1e-3)

        assert e_max.unit == "keV"
        assert_allclose(e_max.value, 4.466836e10, rtol=1e-3)

    def test_verify_npred(self):
        """Veryfing npred is preserved during the stacking"""
        pwl = PowerLawSpectralModel(
            index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )
        self.stacked_dataset.model = pwl

        npred_stacked = self.stacked_dataset.npred().data
        npred_stacked[~self.stacked_dataset.mask_safe] = 0
        npred_summed = np.zeros_like(npred_stacked)

        for obs in self.obs_list:
            obs.model = pwl
            npred_summed[obs.mask_safe] += obs.npred().data[obs.mask_safe]

        assert_allclose(npred_stacked, npred_summed)

    def test_stack_backscal(self):
        """Verify backscal stacking """
        obs1, obs2 = make_observation_list()
        obs1.stack(obs2)
        assert_allclose(obs1.alpha[0], 1.25 / 4.0)
        # When the OFF stack observation counts=0, the alpha is averaged on the total OFF counts for each run.
        assert_allclose(obs1.alpha[1], 2.5 / 8.0)

    def test_stack_gti(self):
        obs1, obs2 = make_observation_list()
        obs1.stack(obs2)
        table_gti = Table({"START": [1.0, 5.0, 14.0], "STOP": [4.0, 8.0, 15.0]})
        table_gti_stacked_obs = obs1.gti.table
        assert_allclose(table_gti_stacked_obs["START"], table_gti["START"])
        assert_allclose(table_gti_stacked_obs["STOP"], table_gti["STOP"])


@requires_data("gammapy-data")
def test_datasets_stack_reduce():
    obs_ids = [23523, 23526, 23559, 23592]
    dataset_list = []
    for obs in obs_ids:
        filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{}.fits"
        ds = SpectrumDatasetOnOff.from_ogip_files(filename.format(obs))
        dataset_list.append(ds)
    datasets = Datasets(dataset_list)
    stacked = datasets.stack_reduce()
    assert_allclose(stacked.livetime.to_value("s"), 6313.8116406202325)

    info_table = datasets.info_table()
    assert_allclose(info_table["n_on"], [124, 126, 119, 90])

    info_table_cum = datasets.info_table(cumulative=True)
    assert_allclose(info_table_cum["n_on"], [124, 250, 369, 459])

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
import numpy as np
from ...utils.testing import requires_data, requires_dependency
from ...utils.random import get_random_state
from ...irf import EffectiveAreaTable, EnergyDispersion
from ...utils.fitting import Fit
from ..models import PowerLaw, ConstantModel, ExponentialCutoffPowerLaw
from ...spectrum import (
    PHACountsSpectrum,
    SpectrumDatasetOnOff
)



class TestSpectrumDatasetOnOff:
    """ Test ON OFF SpectrumDataset"""
    def setup(self):

        etrue = np.logspace(-1,1,10)*u.TeV
        self.e_true = etrue
        ereco = np.logspace(-1,1,5)*u.TeV
        elo = ereco[:-1]
        ehi = ereco[1:]

        self.aeff = EffectiveAreaTable(etrue[:-1],etrue[1:], np.ones(9)*u.cm**2)
        self.edisp = EnergyDispersion.from_diagonal_response(etrue, ereco)

        self.on_counts = PHACountsSpectrum(elo, ehi, np.ones_like(elo), backscal=np.ones_like(elo))
        self.off_counts = PHACountsSpectrum(elo, ehi, np.ones_like(elo)*10, backscal=np.ones_like(elo)*10)

        self.livetime = 1000*u.s

    def test_init_no_model(self):
        dataset = SpectrumDatasetOnOff(counts_on=self.on_counts, counts_off=self.off_counts,
                         aeff=self.aeff, edisp=self.edisp, livetime = self.livetime)

        with pytest.raises(AttributeError):
            dataset.npred()

    def test_alpha(self):
        dataset = SpectrumDatasetOnOff(counts_on=self.on_counts, counts_off=self.off_counts,
                         aeff=self.aeff, edisp=self.edisp, livetime = self.livetime)

        assert dataset.alpha.shape == (4,)
        assert_allclose(dataset.alpha, 0.1)

    def test_npred_no_edisp(self):
        const = 1 / u.TeV / u.cm ** 2 / u.s
        model = ConstantModel(const)
        livetime = 1*u.s
        dataset = SpectrumDatasetOnOff(counts_on=self.on_counts, counts_off=self.off_counts,
                         aeff=self.aeff, model=model, livetime = livetime)

        expected = self.aeff.data.data[0]*(self.aeff.energy.hi[-1]-self.aeff.energy.lo[0])*const*livetime

        assert_allclose(dataset.npred().sum(), expected.value)

@requires_dependency("iminuit")
class TestSimpleFit:
    """Test fit on counts spectra without any IRFs"""

    def setup(self):
        self.nbins = 30
        binning = np.logspace(-1, 1, self.nbins + 1) * u.TeV
        self.source_model = PowerLaw(
            index=2, amplitude=1e5 / u.TeV, reference=0.1 * u.TeV
        )
        self.bkg_model = PowerLaw(
            index=3, amplitude=1e4 / u.TeV, reference=0.1 * u.TeV
        )

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
        obs = SpectrumDatasetOnOff(counts_on=on_vector, counts_off=self.off)
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
        obs1 = SpectrumDatasetOnOff(counts_on=on_vector, counts_off=self.off)
        obs1.model = self.source_model

        src_rebinned = self.src.rebin(2)
        bkg_rebinned = self.off.rebin(2)
        src_rebinned.data.data += self.bkg.rebin(2).data.data

        obs2 = SpectrumDatasetOnOff(counts_on=src_rebinned, counts_off=bkg_rebinned)
        obs2.model = self.source_model

        fit = Fit([obs1, obs2])
        fit.run()
        pars = self.source_model.parameters
        assert_allclose(pars["index"].value, 1.996456, rtol=1e-3)


@requires_data("gammapy-data")
@requires_dependency("iminuit")
class TestSpectralFit:
    """Test fit in astrophysical scenario"""

    def setup(self):
        path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
        obs1 = SpectrumDatasetOnOff.read(path + "pha_obs23523.fits")
        obs2 = SpectrumDatasetOnOff.read(path + "pha_obs23592.fits")
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

    def set_model(self, model ):
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
        assert_allclose(self.obs_list[0].npred()[60], 0.6102, rtol=1e-3)
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


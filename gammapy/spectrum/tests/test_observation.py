# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ...utils.scripts import make_path
from ...irf import EffectiveAreaTable, EnergyDispersion
from .. import (
    PHACountsSpectrum,
    SpectrumObservation,
    SpectrumObservationList,
    models,
    SpectrumSimulation,
)


@requires_dependency("sherpa")
@requires_data("gammapy-data")
def test_spectrum_observation_1():
    """Obs read from file"""
    filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    obs = SpectrumObservation.read(filename)
    pars = dict(
        total_on=189,
        livetime=1581.73681640625 * u.second,
        npred=109.014552,
        excess=169.916667,
        excess_safe_range=116.33,
        lo_threshold=1.0e09,
        hi_threshold=1e11,
    )
    tester = SpectrumObservationTester(obs, pars)
    tester.test_all()


@requires_dependency("sherpa")
@requires_data("gammapy-data")
@pytest.mark.skip
def test_spectrum_observation_2():
    """Simulated obs without background"""
    energy = np.logspace(-2, 2, 100) * u.TeV
    aeff = EffectiveAreaTable.from_parametrization(energy=energy)
    edisp = EnergyDispersion.from_gauss(e_true=energy, e_reco=energy, sigma=0.2, bias=0)
    livetime = 1 * u.h
    source_model = models.PowerLaw(
        index=2.3, amplitude=2.3e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1.4 * u.TeV
    )
    sim = SpectrumSimulation(
        aeff=aeff, edisp=edisp, source_model=source_model, livetime=livetime
    )
    sim.simulate_obs(seed=2309, obs_id=2309)
    obs = sim.obs

    pars = dict(
        total_on=824,
        livetime=livetime,
        npred=292.00223031875987,
        excess=824,
        excess_safe_range=824,
        lo_threshold=0.013219,
        hi_threshold=100,
    )
    tester = SpectrumObservationTester(obs, pars)
    tester.test_all()


@requires_dependency("sherpa")
@requires_data("gammapy-data")
def test_spectrum_observation_3():
    """obs without edisp"""
    energy = np.logspace(-1, 1, 20) * u.TeV
    livetime = 2 * u.h
    on_vector = PHACountsSpectrum(
        energy_lo=energy[:-1], energy_hi=energy[1:], data=np.arange(19), backscal=1
    )
    on_vector.livetime = livetime
    on_vector.obs_id = 2
    aeff = EffectiveAreaTable(
        energy_lo=energy[:-1], energy_hi=energy[1:], data=np.ones(19) * 1e5 * u.m ** 2
    )
    obs = SpectrumObservation(on_vector=on_vector, aeff=aeff)
    pars = dict(
        total_on=171,
        livetime=livetime,
        npred=1425.6,
        excess=171,
        excess_safe_range=171,
        lo_threshold=0.1,
        hi_threshold=10,
    )
    tester = SpectrumObservationTester(obs, pars)
    tester.test_all()


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
    obs1 = SpectrumObservation(
        on_vector=on_vector, off_vector=off_vector1, aeff=aeff, edisp=edisp
    )
    obs2 = SpectrumObservation(
        on_vector=on_vector, off_vector=off_vector2, aeff=aeff, edisp=edisp
    )

    obs_list = [obs1, obs2]
    return obs_list

@pytest.mark.skip
class SpectrumObservationTester:
    def __init__(self, obs, vals):
        self.obs = obs
        self.vals = vals

    @pytest.mark.skip
    def test_all(self):
        self.test_total_stats()
        self.test_stats_in_safe_range()
        self.test_to_sherpa()
        self.test_peek()
        self.test_npred()
        self.test_energy_thresholds()

    def test_npred(self):
        pwl = models.PowerLaw(
            index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )
        npred = self.obs.predicted_counts(model=pwl)
        assert_allclose(npred.total_counts.value, self.vals["npred"], rtol=1e-3)

    @pytest.mark.skip
    def test_total_stats(self):
        excess = self.obs.total_stats.excess
        assert_allclose(excess, self.vals["excess"], atol=1e-3)

    @pytest.mark.skip
    def test_stats_in_safe_range(self):
        stats = self.obs.total_stats_safe_range
        assert_quantity_allclose(stats.energy_min, self.obs.lo_threshold)
        assert_quantity_allclose(stats.energy_max, self.obs.hi_threshold)
        assert_allclose(stats.excess, self.vals["excess_safe_range"], rtol=1e-3)

    @pytest.mark.skip
    @requires_dependency("sherpa")
    def test_to_sherpa(self):
        # This method is not used anywhere but could be useful in the future
        sherpa_obs = self.obs.to_sherpa()
        assert sherpa_obs.counts[10] == self.obs.on_vector.data.data[10].value

    @requires_dependency("matplotlib")
    def test_peek(self):
        with mpl_plot_check():
            self.obs.peek()

    def test_energy_thresholds(self):
        if self.obs.edisp is not None:
            self.obs.compute_energy_threshold(
                method_lo="energy_bias",
                method_hi="none",
                bias_percent_lo=10,
                bias_percent_hi=10,
            )

            assert_allclose(
                self.obs.lo_threshold.value, self.vals["lo_threshold"], rtol=1e-4
            )
            assert_allclose(
                self.obs.hi_threshold.value, self.vals["hi_threshold"], rtol=1e-4
            )
            self.obs.reset_thresholds()


def _read_hess_obs():
    path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
    obs1 = SpectrumObservation.read(path + "pha_obs23523.fits")
    obs2 = SpectrumObservation.read(path + "pha_obs23592.fits")
    return SpectrumObservationList([obs1, obs2])

@requires_data("gammapy-data")
class TestSpectrumObservationList:
    def setup(self):
        self.obs_list = _read_hess_obs()

    def test_write(self, tmpdir):
        self.obs_list.write(outdir=str(tmpdir), pha_typeII=False)
        written_files = make_path(tmpdir).glob("*")
        assert len(list(written_files)) == len(self.obs_list) * 4

        outdir = tmpdir / "pha_typeII"
        self.obs_list.write(outdir=str(outdir), pha_typeII=True)

        test_list = SpectrumObservationList.read(outdir, pha_typeII=True)
        assert str(test_list[0].total_stats) == str(self.obs_list[0].total_stats)

    def test_range(self):
        energy = self.obs_list.safe_range("inclusive")
        assert energy.unit == "TeV"
        assert_allclose(energy.value, [0.891251, 100], rtol=1e-3)

        # TODO: this is not a great test case, should pick two
        # observations where "exclusive" and "inclusive" ranges differ.
        energy = self.obs_list.safe_range("exclusive")
        assert_allclose(energy.value, [0.891251, 100], rtol=1e-3)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
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
    SpectrumObservationStacker,
    models,
    SpectrumSimulation,
)


@requires_dependency("scipy")
@requires_dependency("sherpa")
@requires_data("gammapy-extra")
def test_spectrum_observation_1():
    """Obs read from file"""
    filename = "$GAMMAPY_EXTRA/datasets/joint-crab/spectra/hess/pha_obs23523.fits"
    obs = SpectrumObservation.read(filename)
    pars = dict(
        total_on=189,
        livetime=1581.73681640625 * u.second,
        npred=109.133693,
        excess=169.916667,
        excess_safe_range=117.25,
    )
    tester = SpectrumObservationTester(obs, pars)
    tester.test_all()


@requires_dependency("scipy")
@requires_dependency("sherpa")
@requires_data("gammapy-extra")
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
    )
    tester = SpectrumObservationTester(obs, pars)
    tester.test_all()


@requires_dependency("scipy")
@requires_dependency("sherpa")
@requires_data("gammapy-extra")
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
        total_on=171, livetime=livetime, npred=1425.6, excess=171, excess_safe_range=171
    )
    tester = SpectrumObservationTester(obs, pars)
    tester.test_all()


@requires_dependency("scipy")
@requires_dependency("sherpa")
@requires_data("gammapy-extra")
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


class SpectrumObservationTester:
    def __init__(self, obs, vals):
        self.obs = obs
        self.vals = vals

    def test_all(self):
        self.test_basic()
        self.test_stats_table()
        self.test_total_stats()
        self.test_stats_in_safe_range()
        self.test_to_sherpa()
        self.test_peek()
        self.test_npred()

    def test_basic(self):
        assert "Observation summary report" in str(self.obs)

    def test_npred(self):
        pwl = models.PowerLaw(
            index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )
        npred = self.obs.predicted_counts(model=pwl)
        assert_allclose(npred.total_counts.value, self.vals["npred"])

    def test_stats_table(self):
        table = self.obs.stats_table()
        assert table["n_on"].sum() == self.vals["total_on"]
        assert_quantity_allclose(
            table["livetime"].quantity.max(), self.vals["livetime"]
        )

    def test_total_stats(self):
        excess = self.obs.total_stats.excess
        assert_allclose(excess, self.vals["excess"], atol=1e-3)

    def test_stats_in_safe_range(self):
        stats = self.obs.total_stats_safe_range
        assert_quantity_allclose(stats.energy_min, self.obs.lo_threshold)
        assert_quantity_allclose(stats.energy_max, self.obs.hi_threshold)
        assert_allclose(stats.excess, self.vals["excess_safe_range"], atol=1e-3)

    @requires_dependency("sherpa")
    def test_to_sherpa(self):
        # This method is not used anywhere but could be useful in the future
        sherpa_obs = self.obs.to_sherpa()
        assert sherpa_obs.counts[10] == self.obs.on_vector.data.data[10].value

    @requires_dependency("matplotlib")
    def test_peek(self):
        with mpl_plot_check():
            self.obs.peek()


def _read_hess_obs():
    path = "$GAMMAPY_EXTRA/datasets/joint-crab/spectra/hess/"
    obs1 = SpectrumObservation.read(path + "pha_obs23523.fits")
    obs2 = SpectrumObservation.read(path + "pha_obs23592.fits")
    return SpectrumObservationList([obs1, obs2])


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestSpectrumObservationStacker:
    def setup(self):
        self.obs_list = _read_hess_obs()

        # Change threshold to make stuff more interesting
        self.obs_list.obs(23523).lo_threshold = 1.2 * u.TeV
        self.obs_list.obs(23523).hi_threshold = 50 * u.TeV
        self.obs_list.obs(23592).hi_threshold = 20 * u.TeV
        self.obs_stacker = SpectrumObservationStacker(self.obs_list)
        self.obs_stacker.run()

    def test_basic(self):
        assert "Stacker" in str(self.obs_stacker)
        assert "stacked" in str(self.obs_stacker.stacked_obs.on_vector.phafile)
        counts1 = self.obs_list[0].total_stats_safe_range.n_on
        counts2 = self.obs_list[1].total_stats_safe_range.n_on
        summed_counts = counts1 + counts2
        stacked_counts = self.obs_stacker.stacked_obs.total_stats.n_on
        assert summed_counts == stacked_counts

    def test_thresholds(self):
        energy = self.obs_stacker.stacked_obs.lo_threshold
        assert energy.unit == "keV"
        assert_allclose(energy.value, 8.799e+08, rtol=1e-3)

        energy = self.obs_stacker.stacked_obs.hi_threshold
        assert energy.unit == "keV"
        assert_allclose(energy.value, 4.641e+10, rtol=1e-3)

    def test_verify_npred(self):
        """Veryfing npred is preserved during the stacking"""
        pwl = models.PowerLaw(
            index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )

        npred_stacked = self.obs_stacker.stacked_obs.predicted_counts(model=pwl)

        npred1 = self.obs_list[0].predicted_counts(model=pwl)
        npred2 = self.obs_list[1].predicted_counts(model=pwl)
        # Set npred outside safe range to 0
        npred1.data.data[np.nonzero(self.obs_list[0].on_vector.quality)] = 0
        npred2.data.data[np.nonzero(self.obs_list[1].on_vector.quality)] = 0

        npred_summed = npred1.data.data + npred2.data.data

        assert_allclose(npred_stacked.data.data, npred_summed)

    def test_stack_backscal(self):
        """Verify backscal stacking """
        obs_list = make_observation_list()
        obs_stacker = SpectrumObservationStacker(obs_list)
        obs_stacker.run()
        assert_allclose(obs_stacker.stacked_obs.alpha[0], 1.25 / 4.)
        # When the OFF stack observation counts=0, the alpha is averaged on the total OFF counts for each run.
        assert_allclose(obs_stacker.stacked_obs.alpha[1], 2.5 / 8.)


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestSpectrumObservationList:
    def setup(self):
        self.obs_list = _read_hess_obs()

    def test_stack_method(self):
        obs = self.obs_list.stack()
        assert "Observation summary report" in str(obs)
        assert obs.obs_id == [23523, 23592]

        val = obs.aeff.data.evaluate(energy="1.1 TeV")
        assert val.unit == "cm2"
        assert_allclose(val.value, 1.3466e+09, rtol=1e-3)

        val = obs.edisp.data.evaluate(e_true="1.1 TeV", e_reco="1.3 TeV")
        assert val.unit == ""
        assert_allclose(val.value, 0.06406, rtol=1e-3)

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
        assert_allclose(energy.value, [0.8799, 100], rtol=1e-3)

        # TODO: this is not a great test case, should pick two
        # observations where "exclusive" and "inclusive" ranges differ.
        energy = self.obs_list.safe_range("exclusive")
        assert_allclose(energy.value, [0.8799, 100], rtol=1e-3)

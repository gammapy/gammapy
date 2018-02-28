# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data, mpl_savefig_check
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


@requires_dependency('scipy')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_spectrum_observation_1():
    """Obs read from file"""
    obs = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')
    pars = dict(
        total_on=172,
        livetime=1581.73681640625 * u.second,
        npred=214.55242978860932,
        excess=167.36363636363637,
        excess_safe_range=135,
    )
    tester = SpectrumObservationTester(obs, pars)
    tester.test_all()


@requires_dependency('scipy')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_spectrum_observation_2():
    """Simulated obs without background"""
    energy = np.logspace(-2, 2, 100) * u.TeV
    aeff = EffectiveAreaTable.from_parametrization(energy=energy)
    edisp = EnergyDispersion.from_gauss(e_true=energy, e_reco=energy,
                                        sigma=0.2, bias=0)
    livetime = 1 * u.h
    source_model = models.PowerLaw(index=2.3 * u.Unit(''),
                                   amplitude=2.3e-11 * u.Unit('cm-2 s-1 TeV-1'),
                                   reference=1.4 * u.TeV)
    sim = SpectrumSimulation(aeff=aeff, edisp=edisp, source_model=source_model,
                             livetime=livetime)
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


@requires_dependency('scipy')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_spectrum_observation_3():
    """obs without edisp"""
    energy = np.logspace(-1, 1, 20) * u.TeV
    livetime = 2 * u.h
    on_vector = PHACountsSpectrum(energy_lo=energy[:-1],
                                  energy_hi=energy[1:],
                                  data=np.arange(19),
                                  backscal=1)
    on_vector.livetime = livetime
    on_vector.obs_id = 2
    aeff = EffectiveAreaTable(energy_lo=energy[:-1],
                              energy_hi=energy[1:],
                              data=np.ones(19) * 1e5 * u.m ** 2)
    obs = SpectrumObservation(on_vector=on_vector, aeff=aeff)
    pars = dict(
        total_on=171,
        livetime=livetime,
        npred=1425.6,
        excess=171,
        excess_safe_range=171,
    )
    tester = SpectrumObservationTester(obs, pars)
    tester.test_all()


@requires_dependency('scipy')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
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
    on_vector = PHACountsSpectrum(energy_lo=energy[:-1],
                                  energy_hi=energy[1:],
                                  data=data_on,
                                  backscal=1)
    off_vector1 = PHACountsSpectrum(energy_lo=energy[:-1],
                                    energy_hi=energy[1:],
                                    data=dataoff_1,
                                    backscal=2)
    off_vector2 = PHACountsSpectrum(energy_lo=energy[:-1],
                                    energy_hi=energy[1:],
                                    data=dataoff_2,
                                    backscal=4)
    aeff = EffectiveAreaTable(energy_lo=energy[:-1],
                              energy_hi=energy[1:],
                              data=np.ones(nbin) * 1e5 * u.m ** 2)
    edisp = EnergyDispersion.from_gauss(e_true=energy, e_reco=energy,
                                        sigma=0.2, bias=0)
    on_vector.livetime = livetime
    on_vector.obs_id = 2
    obs1 = SpectrumObservation(on_vector=on_vector, off_vector=off_vector1, aeff=aeff, edisp=edisp)
    obs2 = SpectrumObservation(on_vector=on_vector, off_vector=off_vector2, aeff=aeff, edisp=edisp)

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
        assert 'Observation summary report' in str(self.obs)

    def test_npred(self):
        pwl = models.PowerLaw(index=2 * u.Unit(''),
                              amplitude=2e-11 * u.Unit('cm-2 s-1 TeV-1'),
                              reference=1 * u.TeV)
        npred = self.obs.predicted_counts(model=pwl)
        assert_allclose(npred.total_counts.value, self.vals['npred'])

    def test_stats_table(self):
        table = self.obs.stats_table()
        assert table['n_on'].sum() == self.vals['total_on']
        assert_quantity_allclose(table['livetime'].quantity.max(), self.vals['livetime'])

    def test_total_stats(self):
        excess = self.obs.total_stats.excess
        assert_allclose(excess, self.vals['excess'], atol=1e-3)

    def test_stats_in_safe_range(self):
        stats = self.obs.total_stats_safe_range
        assert_quantity_allclose(stats.energy_min, self.obs.lo_threshold)
        assert_quantity_allclose(stats.energy_max, self.obs.hi_threshold)
        assert_allclose(stats.excess, self.vals['excess_safe_range'], atol=1e-3)

    @requires_dependency('sherpa')
    def test_to_sherpa(self):
        # This method is not used anywhere but could be useful in the future
        sherpa_obs = self.obs.to_sherpa()
        assert sherpa_obs.counts[10] == self.obs.on_vector.data.data[10].value

    @requires_dependency('matplotlib')
    def test_peek(self):
        self.obs.peek()
        mpl_savefig_check()


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumObservationStacker:
    def setup(self):
        self.obs_list = SpectrumObservationList.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha')

        # Change threshold to make stuff more interesting
        self.obs_list.obs(23523).lo_threshold = 1.2 * u.TeV
        self.obs_list.obs(23592).hi_threshold = 20 * u.TeV
        self.obs_list.obs(23523).hi_threshold = 50 * u.TeV
        self.obs_stacker = SpectrumObservationStacker(self.obs_list)
        self.obs_stacker.run()

    def test_basic(self):
        assert 'Stacker' in str(self.obs_stacker)
        counts1 = self.obs_list[0].total_stats_safe_range.n_on
        counts2 = self.obs_list[1].total_stats_safe_range.n_on
        summed_counts = counts1 + counts2
        stacked_counts = self.obs_stacker.stacked_obs.total_stats.n_on
        assert summed_counts == stacked_counts

    def test_thresholds(self):
        actual = self.obs_stacker.stacked_obs.lo_threshold
        desired = 599484250.319 * u.keV
        assert_quantity_allclose(actual, desired)

        actual = self.obs_stacker.stacked_obs.hi_threshold
        desired = 46415888336.1 * u.keV
        assert_quantity_allclose(actual, desired)

    def test_verify_npred(self):
        """Veryfing npred is preserved during the stacking"""
        pwl = models.PowerLaw(index=2 * u.Unit(''),
                              amplitude=2e-11 * u.Unit('cm-2 s-1 TeV-1'),
                              reference=1 * u.TeV)

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


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumObservationList:
    def setup(self):
        self.obs_list = SpectrumObservationList.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha')

    def test_stack_method(self):
        stacked_obs = self.obs_list.stack()
        assert 'Observation summary report' in str(stacked_obs)
        assert stacked_obs.obs_id == [23523, 23592]
        assert_quantity_allclose(stacked_obs.aeff.data.data[10],
                                 86443352.23037884 * u.cm ** 2)
        assert_quantity_allclose(stacked_obs.edisp.data.data[50, 52],
                                 0.027995003769343767)

    def test_write(self, tmpdir):
        self.obs_list.write(outdir=str(tmpdir), pha_typeII=False)
        written_files = make_path(tmpdir).glob('*')
        assert len(list(written_files)) == len(self.obs_list) * 4

        outdir = tmpdir / 'pha_typeII'
        self.obs_list.write(outdir=str(outdir), pha_typeII=True)

        test_list = SpectrumObservationList.read(outdir, pha_typeII=True)
        assert str(test_list[0].total_stats) == str(self.obs_list[0].total_stats)

    def test_range(self):
        erange = self.obs_list.safe_range('inclusive')
        assert_quantity_allclose(erange, [0.5994843, 100] * u.TeV)

        erange = self.obs_list.safe_range('exclusive')
        assert_quantity_allclose(erange, [0.6812921, 100] * u.TeV)

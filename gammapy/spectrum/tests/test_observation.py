# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose, pytest
from ...utils.testing import requires_dependency, requires_data
from ...utils.scripts import make_path
from ...datasets import gammapy_extra
from ...spectrum import (
    SpectrumObservation,
    SpectrumObservationList,
    SpectrumObservationStacker,
    models,
    SpectrumSimulation,
)
from ...irf import EffectiveAreaTable, EnergyDispersion


def get_test_obs():
    test_obs = list()
    if not gammapy_extra.is_available:
        return test_obs
    try:
        import scipy
    except ImportError:
        return test_obs

    # Obs read from file
    obs_1 = SpectrumObservation.read(
        gammapy_extra.filename('datasets/hess-crab4_pha/pha_obs23523.fits'))
    test_obs.append(dict(
        obs=obs_1,
        total_on=172,
        livetime = 1581.73681640625 * u.second,
        excess = 166.428,
        excess_safe_range = 135.428)
    )
    # Simulated obs without background
    energy = np.logspace(-2, 2, 100) * u.TeV
    aeff = EffectiveAreaTable.from_parametrization(energy=energy)
    edisp = EnergyDispersion.from_gauss(e_true = energy, e_reco = energy)
    livetime = 1 * u.h
    source_model = models.PowerLaw(index = 2.3 * u.Unit(''),
                                   amplitude = 2.3e-11 * u.Unit('cm-2 s-1 TeV-1'),
                                   reference = 1.4 * u.TeV)
    sim = SpectrumSimulation(aeff=aeff, edisp=edisp, source_model=source_model,
                             livetime=livetime)
    sim.simulate_obs(seed=2309, obs_id=2309)
    test_obs.append(dict(
        obs=sim.obs,
        total_on=821,
        livetime = livetime, 
        excess = 821, 
        excess_safe_range = 821)
    )
    return test_obs

@pytest.mark.parametrize('obs', get_test_obs())
@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_spectrum_observation(obs):
    tester=SpectrumObservationTester(obs)
    tester.test_all()

class SpectrumObservationTester:
    def __init__(self, obs_dict):
        self.obs=obs_dict.pop('obs')
        self.vals=obs_dict

    def test_all(self):
        self.test_basic()
        self.test_stats_table()
        self.test_total_stats()
        self.test_stats_in_safe_range()
        self.test_peek()

    def test_basic(self):
        assert 'Observation summary report' in str(self.obs)

    def test_stats_table(self):
        table=self.obs.stats_table()
        assert table['n_on'].sum() == self.vals['total_on']
        assert_quantity_allclose(table['livetime'].max(), self.vals['livetime'])

    def test_total_stats(self):
        excess=self.obs.total_stats.excess
        assert_allclose(excess, self.vals['excess'], atol=1e-3)

    def test_stats_in_safe_range(self):
        stats=self.obs.total_stats_safe_range
        assert_quantity_allclose(stats.energy_min, self.obs.lo_threshold)
        assert_quantity_allclose(stats.energy_max, self.obs.hi_threshold)
        assert_allclose(stats.excess, self.vals['excess_safe_range'], atol=1e-3)

    @requires_dependency('matplotlib')
    def test_peek(self):
        self.obs.peek()


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumObservationStacker:
    def setup(self):
        self.obs_list=SpectrumObservationList.read(
            '$GAMMAPY_EXTRA/datasets/hess-crab4_pha')

        # Change threshold to make stuff more interesting
        self.obs_list.obs(23523).lo_threshold=1.2 * u.TeV
        self.obs_stacker=SpectrumObservationStacker(self.obs_list)
        self.obs_stacker.run()

    def test_basic(self):
        assert 'Stacker' in str(self.obs_stacker)
        counts1=self.obs_list[0].total_stats_safe_range.n_on
        counts2=self.obs_list[1].total_stats_safe_range.n_on
        summed_counts=counts1 + counts2
        stacked_counts=self.obs_stacker.stacked_obs.total_stats.n_on
        assert summed_counts == stacked_counts

    def test_verify_npred(self):
        """Veryfing npred is preserved during the stacking"""
        pwl=models.PowerLaw(index=2 * u.Unit(''),
                              amplitude=2e-11 * u.Unit('cm-2 s-1 TeV-1'),
                              reference=1 * u.TeV)

        npred_stacked=self.obs_stacker.stacked_obs.predicted_counts(model=pwl)

        npred1=self.obs_list[0].predicted_counts(model=pwl)
        npred2=self.obs_list[1].predicted_counts(model=pwl)
        # Set npred outside safe range to 0
        npred1.data[np.nonzero(self.obs_list[0].on_vector.quality)]=0
        npred2.data[np.nonzero(self.obs_list[1].on_vector.quality)]=0

        npred_summed=npred1.data + npred2.data

        assert_allclose(npred_stacked.data, npred_summed)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumObservationList:
    def setup(self):
        self.obs_list=SpectrumObservationList.read(
            '$GAMMAPY_EXTRA/datasets/hess-crab4_pha')

    def test_stack_method(self):
        stacked_obs=self.obs_list.stack()
        assert 'Observation summary report' in str(stacked_obs)
        assert stacked_obs.obs_id == [23523, 23592]
        assert_quantity_allclose(stacked_obs.aeff.data[10], 86443352.23037884 * u.cm ** 2)
        assert_quantity_allclose(stacked_obs.edisp.data[50, 52], 0.029627067949207702)

    def test_write(self, tmpdir):
        self.obs_list.write(outdir=str(tmpdir), pha_typeII=False)
        written_files=make_path(tmpdir).glob('*')
        assert len(list(written_files)) == len(self.obs_list) * 4

        outdir=tmpdir / 'pha_typeII'
        self.obs_list.write(outdir=str(outdir), pha_typeII=True)

        test_list=SpectrumObservationList.read(outdir, pha_typeII=True)
        assert str(test_list[0].total_stats) == str(self.obs_list[0].total_stats)

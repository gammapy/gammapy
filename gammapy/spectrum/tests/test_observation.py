# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...spectrum import (
    SpectrumObservation,
    SpectrumObservationList,
    SpectrumObservationStacker,
    models
)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumObservation:
    def setup(self):
        self.obs = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')

    def test_stats_table(self):
        table = self.obs.stats_table()
        assert table['n_on'].sum() == 172
        assert_quantity_allclose(table['livetime'].max(), 1581.73681640625 * u.second)

    def test_total_stats(self):
        excess = self.obs.total_stats.excess
        assert_allclose(excess, 166.428, atol=1e-3)

    def test_stats_in_safe_range(self):
        stats = self.obs.total_stats_safe_range
        assert_quantity_allclose(stats.energy_min, self.obs.lo_threshold)
        assert_quantity_allclose(stats.energy_max, self.obs.hi_threshold)
        assert_allclose(stats.excess, 135.428, atol=1e-3)

    @requires_dependency('matplotlib')
    def test_peek(self):
        self.obs.peek()


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumObservationStacker:
    def setup(self):
        self.obs_list = SpectrumObservationList.read(
            '$GAMMAPY_EXTRA/datasets/hess-crab4_pha')

        # Change threshold to make stuff more interesting
        self.obs_list.obs(23523).lo_threshold = 1.2 * u.TeV
        self.obs_stacker = SpectrumObservationStacker(self.obs_list)
        self.obs_stacker.run()

    def test_basic(self):
        assert 'Stacker' in str(self.obs_stacker)
        counts1 = self.obs_list[0].total_stats_safe_range.n_on
        counts2 = self.obs_list[1].total_stats_safe_range.n_on
        summed_counts = counts1 + counts2
        stacked_counts = self.obs_stacker.stacked_obs.total_stats.n_on
        assert summed_counts == stacked_counts

    def test_verify_npred(self):
        """Veryfing npred is preserved during the stacking"""
        pwl = models.PowerLaw(index=2 * u.Unit(''),
                              amplitude=2e-11 * u.Unit('cm-2 s-1 TeV-1'),
                              reference=1 * u.TeV)

        npred_stacked = self.obs_stacker.stacked_obs.predicted_counts(model=pwl)

        npred1 = self.obs_list[0].predicted_counts(model=pwl)
        npred2 = self.obs_list[1].predicted_counts(model=pwl)
        # Set npred outside safe range to 0
        npred1.data[np.nonzero(self.obs_list[0].on_vector.quality)] = 0
        npred2.data[np.nonzero(self.obs_list[1].on_vector.quality)] = 0

        npred_summed = npred1.data + npred2.data

        assert_allclose(npred_stacked.data, npred_summed)

    def test_stack_method_on_list(self):
        stacked_obs = self.obs_list.stack()
        assert 'Observation summary report' in str(stacked_obs)
        assert stacked_obs.obs_id == [23523, 23592]
        assert_quantity_allclose(stacked_obs.aeff.data[10],86443352.23037884*u.cm**2)
        assert_quantity_allclose(stacked_obs.edisp.data[50,52],0.029627067949207702)
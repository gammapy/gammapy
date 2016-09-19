# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...spectrum import SpectrumObservation, models


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumObservation:
    def setup(self):
        self.obs = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')
        self.obs2 = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23592.fits')

        # Change threshold to make stuff more interesting
        self.obs.lo_threshold = 1.2 * u.TeV

        self.obs_stack = SpectrumObservation.stack([self.obs, self.obs2])

    def test_stack(self):
        # Veryfing npred is preserved during the stacking
        pwl = models.PowerLaw(index=2 * u.Unit(''),
                              amplitude=2e-11 * u.Unit('cm-2 s-1 TeV-1'),
                              reference=1 * u.TeV)

        npred1 = self.obs.predicted_counts(model=pwl)
        npred2 = self.obs2.predicted_counts(model=pwl)
        npred_stacked = self.obs_stack.predicted_counts(model=pwl)

        # Set npred outside safe range to 0
        npred1.data[np.nonzero(self.obs.on_vector.quality)] = 0
        npred2.data[np.nonzero(self.obs2.on_vector.quality)] = 0

        npred_summed = npred1 + npred2

        assert_allclose(npred_stacked.data, npred_summed.data)

    def test_stats_table(self):
        table = self.obs.stats_table()
        assert table['n_on'].sum() == 172
        assert_quantity_allclose(table['livetime'].max(), 1581.73681640625 * u.second)

    @requires_dependency('matplotlib')
    def test_peek(self):
        self.obs.peek()

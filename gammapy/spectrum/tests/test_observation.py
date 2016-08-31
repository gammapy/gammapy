# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, assert_quantity_allclose
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
from ...spectrum import SpectrumObservation, models

@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
@requires_dependency('scipy')
def test_spectrum_observation():
    phafile = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23523.fits")
    obs = SpectrumObservation.read(phafile)
    obs.peek()


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_observation_stacking():
    obs1 = SpectrumObservation.read(
        '$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')
    obs2 = SpectrumObservation.read(
        '$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23592.fits')

    # Change threshold to make stuff more interesing
    obs1.on_vector.lo_threshold = 1.2 * u.TeV

    stacked_obs = SpectrumObservation.stack([obs1, obs2])

    # Veryfing npred is preserved during the stacking
    pwl = models.PowerLaw(index=2 * u.Unit(''),
                          amplitude=2e-11 * u.Unit('cm-2 s-1 TeV-1'),
                          reference=1 * u.TeV)

    npred1 = obs1.predicted_counts(model=pwl)
    npred2 = obs2.predicted_counts(model=pwl)
    npred_stacked = stacked_obs.predicted_counts(model=pwl)

    # Set npred outside safe range to 0
    npred1.data[np.nonzero(obs1.on_vector.quality)] = 0
    npred2.data[np.nonzero(obs2.on_vector.quality)] = 0

    npred_summed = npred1 + npred2

    assert_allclose(npred_stacked.data, npred_summed.data)

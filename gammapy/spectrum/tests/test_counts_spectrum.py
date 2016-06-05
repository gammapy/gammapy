# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import astropy.units as u
from astropy.tests.helper import pytest
from numpy.testing import assert_equal, assert_allclose

from .. import CountsSpectrum, SpectrumExtraction, SpectrumFitResult, \
    SpectrumObservation

from ...data import ObservationTable
from ...datasets import gammapy_extra
from ...utils.testing import requires_data, requires_dependency
from ...utils.energy import EnergyBounds


@requires_data('gammapy-extra')
def test_CountsSpectrum(tmpdir):
    # create from scratch
    counts = [0, 0, 2, 5, 17, 3] * u.ct
    bins = EnergyBounds.equal_log_spacing(1, 10, 7, 'TeV')
    actual = False
    try:
        spec = CountsSpectrum(data=counts, energy=bins)
    except(ValueError):
        actual = True
    desired = True
    assert_equal(actual, desired)

    bins = EnergyBounds.equal_log_spacing(1, 10, 6, 'TeV')
    actual = False
    try:
        spec = CountsSpectrum(data=counts, energy=bins)
    except(ValueError):
        actual = True
    desired = False
    assert_equal(actual, desired)
    
    test_e = bins[2] + 0.1 * u.TeV
    test_eval = spec.evaluate(energy=test_e, method='nearest')
    assert_allclose(test_eval, spec.data[2])

    # Test I/O 
    f = tmpdir / 'test.fits'
    spec.write(f)
    spec2 = CountsSpectrum.read(f)

    assert (spec.energy.data == spec2.energy.data).all()

    #add two spectra
    bins = spec.energy.nbins
    counts = np.array(np.random.rand(bins) * 10, dtype=int) * u.ct
    pha2 = CountsSpectrum(data=counts, energy=spec.energy)
    pha_sum = np.sum([spec, pha2])
    desired = spec.data[5] + counts[5]
    actual = pha_sum.data[5]
    assert_equal(actual, desired)

@pytest.mark.xfail(reason='broken')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_n_pred():
    fitresult = gammapy_extra.filename(
        'test_datasets/spectrum/fit_result_PowerLaw_reference.yaml')

    obs_id = [23523, 23592]
    filenames = [gammapy_extra.filename('datasets/hess-crab4_pha/pha_obs{}.fits'.format(
        _)) for _ in obs_id]
    obs = [SpectrumObservation.read(_) for _ in filenames]
    fit = SpectrumFitResult.from_yaml(fitresult)

    n_pred_vec = [CountsSpectrum.get_npred(fit, o) for o in obs]
    n_pred = np.sum(n_pred_vec)

    assert_allclose(max(n_pred.data).value, 52.5, atol=0.1)

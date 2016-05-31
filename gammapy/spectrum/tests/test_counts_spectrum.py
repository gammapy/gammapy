# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from .. import CountsSpectrum, SpectrumExtraction, SpectrumFitResult, \
    SpectrumObservationList

from ...data import ObservationTable
from ...datasets import gammapy_extra
from ...utils.testing import requires_data, requires_dependency
from ...utils.energy import EnergyBounds


@requires_data('gammapy-extra')
def test_CountsSpectrum():
    # create from scratch
    counts = [0, 0, 2, 5, 17, 3]
    bins = EnergyBounds.equal_log_spacing(1, 10, 7, 'TeV')
    actual = False
    try:
        spec = CountsSpectrum(counts, bins)
    except(ValueError):
        actual = True
    desired = True
    assert_equal(actual, desired)

    bins = EnergyBounds.equal_log_spacing(1, 10, 6, 'TeV')
    actual = False
    try:
        spec = CountsSpectrum(counts, bins)
    except(ValueError):
        actual = True
    desired = False
    assert_equal(actual, desired)

    # set backscal
    spec.backscal = 15

    # test to_fits
    spec.to_fits()

    # Read pha file
    f = gammapy_extra.filename('datasets/hess-crab4_pha/ogip_data/pha_run23526.fits')
    pha1 = CountsSpectrum.read_pha(f)

    #add two spectra
    bins = pha1.energy_bounds.nbins
    counts = np.array(np.random.rand(bins) * 10, dtype=int)
    pha2 = CountsSpectrum(counts, pha1.energy_bounds)
    pha_sum = np.sum([pha1, pha2])
    desired = pha1.counts[5] + counts[5]
    actual = pha_sum.counts[5]
    assert_equal(actual, desired)


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_n_pred():
    fitresult = gammapy_extra.filename(
        'test_datasets/spectrum/fit_result_PowerLaw_reference.yaml')

    obs_table_file = gammapy_extra.filename(
        'datasets/hess-crab4_pha/observation_table.fits')

    obs_table = ObservationTable.read(obs_table_file)
    obs = SpectrumObservationList.from_observation_table(obs_table)
    fit = SpectrumFitResult.from_yaml(fitresult)

    n_pred_vec = [CountsSpectrum.get_npred(fit, o) for o in obs]
    n_pred = np.sum(n_pred_vec)

    assert_allclose(max(n_pred.counts), 52.5, atol=0.1)

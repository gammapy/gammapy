# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from numpy.testing import assert_equal

from gammapy.datasets import gammapy_extra
from gammapy.utils.testing import requires_data
from .. import CountsSpectrum
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
    f = gammapy_extra.filename('datasets/hess-crab4_pha/pha_run23526.pha')
    pha1 = CountsSpectrum.read(f)

    #add two spectra
    bins = pha1.energy_bounds.nbins
    counts = np.array(np.random.rand(bins)*10 , dtype = int)
    pha2 = CountsSpectrum(counts, pha1.energy_bounds)
    pha_sum = np.sum([pha1, pha2])
    desired = pha1.counts[5] + counts[5]
    actual = pha_sum.counts[5]
    assert_equal(actual, desired)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_equal
from .. import CountsSpectrum
from ...utils.energy import EnergyBounds


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

    #test to_fits
    spec.to_fits()

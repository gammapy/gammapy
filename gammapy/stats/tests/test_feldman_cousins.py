# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ...stats import (
    fc_find_acceptance_region_gauss,
    fc_find_acceptance_region_poisson,
    fc_construct_acceptance_intervals_pdfs,
    fc_get_upper_and_lower_limit,
    fc_fix_upper_and_lower_limit,
    fc_find_limit,
    fc_find_average_upper_limit,
    fc_construct_acceptance_intervals,
)


@pytest.mark.skipif('not HAS_SCIPY')
def test_acceptance_region_gauss():

    fSigma  = 1
    fNSigma = 10
    fNStep  = 1000
    fCL     = 0.90

    XBins  = np.linspace(-fNSigma*fSigma, fNSigma*fSigma, fNStep, endpoint=True)

    # The test reverses a result from the Feldman and Cousins paper. According
    # to Table X, for a measured value of 2.6 the 90% confidence interval should
    # be 1.02 and 4.24. Reversed that means that for mu=1.02, the acceptance
    # interval should end at 2.6 and for mu=4.24 should start at 2.6.
    (x_min, x_max) = fc_find_acceptance_region_gauss(1.02, fSigma, XBins, fCL)
    assert_allclose(x_max, 2.6, 0.1)

    (x_min, x_max) = fc_find_acceptance_region_gauss(4.24, fSigma, XBins, fCL)
    assert_allclose(x_min, 2.6, 0.1)

    # At mu=0, the confidence interval should start at the negative XBins range.
    (x_min, x_max) = fc_find_acceptance_region_gauss(0, fSigma, XBins, fCL)
    assert_allclose(x_min, -fNSigma*fSigma)


@pytest.mark.skipif('not HAS_SCIPY')
def test_acceptance_region_poisson():

    fBackground  = 0.5
    fNBinsX      = 100
    fCL          = 0.90

    XBins  = np.arange(0, fNBinsX)

    # The test reverses a result from the Feldman and Cousins paper. According
    # to Table IV, for a measured value of 10 the 90% confidence interval should
    # be 5.00 and 16.00. Reversed that means that for mu=5.0, the acceptance
    # interval should end at 10 and for mu=16.00 should start at 10.
    (x_min, x_max) = fc_find_acceptance_region_poisson(5.00, fBackground, XBins, fCL)
    assert_allclose(x_max, 10)

    (x_min, x_max) = fc_find_acceptance_region_poisson(16.00, fBackground, XBins, fCL)
    assert_allclose(x_min, 10)


@pytest.mark.skipif('not HAS_SCIPY')
def test_numerical_confidence_interval_pdfs():

    from scipy import stats

    fBackground  = 3.0
    fStepWidthMu = 0.005
    fMuMin       = 0
    fMuMax       = 15
    fNBinsX      = 50
    fCL          = 0.90

    XBins  = np.arange(0, fNBinsX)
    MuBins = np.linspace(fMuMin, fMuMax, fMuMax/fStepWidthMu + 1, endpoint=True)

    Matrix = [stats.poisson(mu+fBackground).pmf(XBins) for mu in MuBins]

    AcceptanceIntervals = fc_construct_acceptance_intervals_pdfs(Matrix, fCL)

    UpperLimitNum, LowerLimitNum, _ = fc_get_upper_and_lower_limit(MuBins, XBins, AcceptanceIntervals)

    fc_fix_upper_and_lower_limit(UpperLimitNum, LowerLimitNum)

    x_measured = 6
    upper_limit = fc_find_limit(x_measured, UpperLimitNum, MuBins)
    lower_limit = fc_find_limit(x_measured, LowerLimitNum, MuBins)

    # Values are taken from Table IV in the Feldman and Cousins paper.
    assert_allclose(upper_limit, 8.47, 0.01)
    assert_allclose(lower_limit, 0.15, 0.01)

    average_upper_limit = fc_find_average_upper_limit(XBins,Matrix, UpperLimitNum, MuBins)

    # Values are taken from Table XII in the Feldman and Cousins paper.
    # A higher accuracy would require a higher fMuMax, which would increase
    # the computation time.
    assert_allclose(average_upper_limit, 4.4, 0.1)


@pytest.mark.skipif('not HAS_SCIPY')
def test_numerical_confidence_interval_values():

    from scipy import stats

    fBackground  = 3.0
    fStepWidthMu = 0.005
    fMuMin       = 0
    fMuMax       = 15
    fNBinsX      = 50
    fCL          = 0.90

    XBins  = np.arange(0, fNBinsX)
    MuBins = np.linspace(fMuMin, fMuMax, fMuMax/fStepWidthMu + 1, endpoint=True)

    distribution_dict = dict((mu, [stats.poisson.rvs(mu+fBackground,size=2000)]) for mu in MuBins)

    AcceptanceIntervals = fc_construct_acceptance_intervals(distribution_dict, XBins, fCL)

    UpperLimitNum, LowerLimitNum, _ = fc_get_upper_and_lower_limit(MuBins, XBins, AcceptanceIntervals)

    fc_fix_upper_and_lower_limit(UpperLimitNum, LowerLimitNum)

    x_measured = 6
    upper_limit = fc_find_limit(x_measured, UpperLimitNum, MuBins)

    # Value taken from Table IV in the Feldman and Cousins paper.
    assert_allclose(upper_limit, 8.5, 0.1)

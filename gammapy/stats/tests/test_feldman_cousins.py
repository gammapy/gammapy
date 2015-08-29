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
    fc_find_confidence_interval_gauss,
    fc_find_confidence_interval_poisson,
    fc_construct_confidence_belt_pdfs,
    fc_get_upper_and_lower_limit,
    fc_fix_upper_and_lower_limit,
    fc_find_limit,
    fc_find_average_upper_limit,
    fc_construct_confidence_belt,
)


@pytest.mark.skipif('not HAS_SCIPY')
def test_confidence_interval_gauss():

    fSigma = 1
    fMuMin  = 0
    fMuMax  = 8
    fMuStep = 100
    fNSigma = 10
    fNStep  = 1000
    fCL     = 0.90

    XBins  = np.linspace(-fNSigma*fSigma, fNSigma*fSigma, fNStep, endpoint=True)
    MuBins = np.linspace(fMuMin, fMuMax, fMuStep, endpoint=True)

    LowerLimit, UpperLimit = fc_find_confidence_interval_gauss(MuBins[-1], fSigma, XBins, fCL)
    assert_allclose(LowerLimit, 6.356356356356358)
    assert_allclose(UpperLimit, 9.65965965965966)


@pytest.mark.skipif('not HAS_SCIPY')
def test_confidence_interval_poisson():

    fBackground  = 3.0
    fStepWidthMu = 0.005
    fMuMin       = 0
    fMuMax       = 50
    fNBinsX      = 100
    fCL          = 0.90

    XBins  = np.arange(0, fNBinsX)
    MuBins = np.linspace(fMuMin, fMuMax, fMuMax/fStepWidthMu + 1, endpoint=True)

    LowerLimit, UpperLimit = fc_find_confidence_interval_poisson(MuBins[-1], fBackground, XBins, fCL)
    assert_allclose(LowerLimit, 42)
    assert_allclose(UpperLimit, 66)

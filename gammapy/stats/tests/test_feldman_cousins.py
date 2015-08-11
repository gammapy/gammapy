# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest

from ...stats import (
    find_confidence_interval_gauss,
    find_confidence_interval_poisson,
    construct_confidence_belt_PDFs,
    get_upper_and_lower_limit,
    fix_upper_and_lower_limit,
    find_limit,
    find_average_upper_limit,
    construct_confidence_belt,
)

def test_significance_to_probability_normal():

  sigma = 1

  fMuMin  = 0
  fMuMax  = 8
  fMuStep = 100
  fNSigma = 10
  fNStep  = 1000
  fCL = 0.90

  x_bins  = numpy.linspace(-fNSigma*sigma, fNSigma*sigma, fNStep, endpoint=True)
  mu_bins = numpy.linspace(fMuMin, fMuMax, fMuStep, endpoint=True)

  lower_limit, upper_limit = find_confidence_interval_gauss(mu_bins[-1], sigma, x_bins, fCL)
  assert_allclose(lower_limit, 6.356)
  assert_allclose(upper_limit, 9.660)
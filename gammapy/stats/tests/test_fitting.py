# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.modeling.models import Gaussian1D
from ...stats import PoissonLikelihoodFitter

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.xfail
@pytest.mark.skipif('not HAS_SCIPY')
def test_PoissonLikelihoodFitter():
    model = Gaussian1D(amplitude=1000, mean=2, stddev=3)
    x = np.arange(-10, 20, 0.1)
    dx = 0.1 * np.ones_like(x)
    # initialise random number generator
    random_state = np.random.RandomState(seed=0)
    y = random_state.poisson(dx * model(x))

    fitter = PoissonLikelihoodFitter()
    model = fitter(model, x, y, dx)
    expected = [995.29239606, 1.99019548, 3.00869128]
    assert_allclose(model.parameters, expected, rtol=1e-3)

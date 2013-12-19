# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.modeling.models import Gaussian1D
from ..fitting import PoissonLikelihoodFitter

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_PoissonLikelihoodFitter():
    model = Gaussian1D(amplitude=1000, mean=2, stddev=3)
    x = np.arange(-10, 20, 0.1)
    dx = 0.1 * np.ones_like(x)
    np.random.seed(0)
    y = np.random.poisson(dx * model(x))

    fitter = PoissonLikelihoodFitter()
    model = fitter(model, x, y, dx)
    assert_allclose(model.parameters, [995.29239606, 1.99019548, 3.00869128], rtol=1e-3)

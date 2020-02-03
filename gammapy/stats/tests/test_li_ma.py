# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.stats import (
    lm_loglikelihood,
    lm_dexcess_down,
    lm_dexcess_up,
    lm_dexcess,
    lm_significance
)


def test_significance():
    n_on = 10
    n_off = 20
    alpha = 0.1
    assert_allclose(lm_significance(n_on, n_off, alpha), 3.6850322319420274, atol=0.01)

    n_off = [20, 8]
    alpha = [0.1, 1]
    sol = [3.6850322319420274, 0.47189166410776]
    assert_allclose(lm_significance(n_on, n_off, alpha), sol, atol=0.01)


def test_error():
    n_on = 10
    n_off = 200
    alpha = 0.1
    assert_allclose(lm_dexcess_up(n_on, n_off, alpha), 3.7529, atol=0.01)
    assert_allclose(lm_dexcess_down(n_on, n_off, alpha), 3.2104, atol=0.01)
    assert_allclose(lm_dexcess(n_on, n_off, alpha), 3.4816, atol=0.01)

    n_on = [50, 7.]
    n_off = [20, 8]
    alpha = [0.1, 1]
    sol = [7.0893, 3.9353]
    assert_allclose(lm_dexcess(n_on, n_off, alpha), sol, atol=0.01)

    n_on = 7
    n_off = 8
    alpha = 1.
    assert_allclose(lm_dexcess_up(n_on, n_off, alpha), 3.9140, atol=0.01)
    assert_allclose(lm_dexcess_down(n_on, n_off, alpha), 3.9565, atol=0.01)

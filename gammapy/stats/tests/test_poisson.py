# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.stats import (
    excess_matching_significance,
    excess_matching_significance_on_off,
    excess_ul_helene,
)


def test_excess_ul_helene():
    # The reference values here are from the HESS software
    # TODO: change to reference values from the Helene paper
    excess = excess_ul_helene(excess=50, excess_error=40, significance=3)
    assert_allclose(excess, 171.353908, rtol=1e-3)

    excess = excess_ul_helene(excess=10, excess_error=6, significance=2)
    assert_allclose(excess, 22.123334, rtol=1e-3)

    excess = excess_ul_helene(excess=-23, excess_error=8, significance=3)
    assert_allclose(excess, 13.372179, rtol=1e-3)

    # Check in the very high, Gaussian signal limit, where you have
    # 10000 photons with Poisson noise and no background.
    excess = excess_ul_helene(excess=10000, excess_error=100, significance=1)
    assert_allclose(excess, 10100, atol=0.1)



#

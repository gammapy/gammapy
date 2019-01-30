# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from ...stats import (
    significance_to_probability_normal,
    probability_to_significance_normal,
    probability_to_significance_normal_limit,
    significance_to_probability_normal_limit,
)


def test_significance_to_probability_normal():
    significance = 5
    p = significance_to_probability_normal(significance)
    assert_allclose(p, 2.8665157187919328e-07)

    s = probability_to_significance_normal(p)
    assert_allclose(s, significance)


def test_significance_to_probability_normal_limit():
    significance = 5
    p = significance_to_probability_normal_limit(significance)
    assert_allclose(p, 2.792513e-07)

    s = probability_to_significance_normal_limit(p)
    assert_allclose(s, significance)

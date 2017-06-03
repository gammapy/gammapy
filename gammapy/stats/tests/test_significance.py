# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import pytest
from ...utils.testing import requires_dependency
from ...stats import (
    convert_likelihood,
    significance_to_probability_normal,
    probability_to_significance_normal,
    probability_to_significance_normal_limit,
    significance_to_probability_normal_limit,
)


@requires_dependency('scipy')
def test_significance_to_probability_normal():
    significance = 5
    p = significance_to_probability_normal(significance)
    assert_allclose(p, 2.8665157187919328e-07)

    s = probability_to_significance_normal(p)
    assert_allclose(s, significance)


@requires_dependency('scipy')
def test_significance_to_probability_normal_limit():
    significance = 5
    p = significance_to_probability_normal_limit(significance)
    assert_allclose(p, 2.792513e-07)

    s = probability_to_significance_normal_limit(p)
    assert_allclose(s, significance)


@requires_dependency('scipy')
def test_convert_likelihood_examples():
    """Check a few example values and conversions."""
    assert_allclose(convert_likelihood(to='probability', significance=3), 0.0013498980316300933)
    assert_allclose(convert_likelihood(to='probability', significance=5), 2.8665157187919333e-07)

    assert_allclose(convert_likelihood(to='probability', ts=30, df=1), 4.3204630578274955e-08)
    assert_allclose(convert_likelihood(to='probability', ts=30, df=4), 4.894437128029217e-06)

    # TODO: activate the chi2 tests once that's fixed ...
    # Here's a few examples from the table
    # "Upper-tail critical values of chi-square distribution"
    # http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    # assert_allclose(convert_likelihood(to='chi2', probability=0.1, df=1), 2.706)
    # assert_allclose(convert_likelihood(to='chi2', probability=0.1, df=10), 15.987)
    # assert_allclose(convert_likelihood(to='chi2', probability=0.01, df=1), 6.635)
    # assert_allclose(convert_likelihood(to='chi2', probability=0.01, df=1), 23.209)

    # For `df=1` the simple relation `significance=sqrt(ts)` should roughly hold
    # in the large `ts` limit (for ts ~ 1 actually significance ~ 0.47, not significance = 1).
    ts = [100, 900]
    significance_expected = np.sqrt(ts)
    significance_actual = convert_likelihood(to='significance', ts=ts, df=1)
    # Output: significance_actual = [9.931126, 29.976912]
    assert_allclose(significance_actual, significance_expected, rtol=1e-2)


@requires_dependency('scipy')
def test_convert_likelihood_invalid_input():
    """Check that invalid input raises an exception."""
    with pytest.raises(ValueError) as err:
        convert_likelihood(to='aaa', ts=30, df=4)
    assert "Invalid parameter `to`: aaa" in str(err)

    with pytest.raises(ValueError) as err:
        convert_likelihood(to='ts')
    assert "You have to pass exactly one" in str(err)

    with pytest.raises(ValueError) as err:
        convert_likelihood(to='ts', ts=99, chi2=99)
    assert "You have to pass exactly one" in str(err)

    with pytest.raises(ValueError) as err:
        convert_likelihood(to='significance', ts=99)
    assert 'You have to specify the number of degrees of freedom via the `df` parameter.' in str(err)


@requires_dependency('scipy')
def test_convert_likelihood_roundtrip():
    """Check that round-tripping works for each pair of quantities."""
    significance = [1, 3, 5, 10]

    for df in [1, 2, 3, 4]:
        # TODO: add tests for `chi2`
        for to in ['probability', 'ts']:
            val = convert_likelihood(to=to, significance=significance, df=df)
            significance2 = convert_likelihood(to='significance', df=df, **{to: val})
            assert_allclose(significance2, significance)

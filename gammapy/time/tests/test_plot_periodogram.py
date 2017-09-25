# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ..period import lomb_scargle
from ...utils.testing import requires_dependency
from ..plot_periodogram import plot_periodogram
from .test_period import simulate_test_data


@requires_dependency('scipy')
@pytest.mark.parametrize('test_case', [
    dict(period=7, amplitude=2, t_length=100, n_data=1000,
         n_obs=500, n_outliers=0, dt=0.5,
         max_period=None, criteria='all', n_bootstraps=10),
])
def test_lomb_scargle_plot(test_case):
    test_data = simulate_test_data(
        test_case['period'], test_case['amplitude'], test_case['t_length'],
        test_case['n_data'], test_case['n_obs'], test_case['n_outliers'],
    )
    result = lomb_scargle(
        test_data['t'], test_data['y'], test_data['dy'], test_case['dt'],
        test_case['max_period'], test_case['criteria'], test_case['n_bootstraps'],
    )
    plot_periodogram(
        test_data['t'], test_data['y'], test_data['dy'], result['pgrid'],
        result['psd'], result['swf'], result['period'], result['fap'],
    )

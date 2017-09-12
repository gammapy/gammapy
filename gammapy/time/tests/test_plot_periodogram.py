import pytest
from ..lomb_scargle import lomb_scargle
from ..plot_periodogram import plot_periodogram
from .test_lomb_scargle import simulate_test_data


@pytest.mark.parametrize('test_case', [
    dict(period=7, amplitude=2, t_length=100, n_data=1000,
         n_observations=1000 / 2, n_outliers=0, dt=0.1,
         max_period=10, criteria='boot', n_bootstraps=100),
    dict(period=7, amplitude=2, t_length=100, n_data=1000,
         n_observations=1000 / 2, n_outliers=0, dt=0.1,
         max_period='None', criteria='None', n_bootstraps='None'),
])
def test_lomb_scargle_plot(test_case):
    test_data = simulate_test_data(
        test_case['period'], test_case['amplitude'], test_case['t_length'],
        test_case['n_data'], test_case['n_observations'], test_case['n_outliers']
    )
    result = lomb_scargle(
        test_data['t'], test_data['y'], test_data['dy'], test_case['dt'],
        test_case['max_period'], test_case['criteria'], test_case['n_bootstraps'],
    )
    plot_periodogram(
        test_data['t'], test_data['y'], test_data['dy'], result['pgrid'],
        result['psd'], result['swf'], result['period'],
        result['significance']
    )

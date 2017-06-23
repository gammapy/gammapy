import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..lomb_scargle import lomb_scargle, lomb_scargle_plot

def simulate_test_data(period, amplitude, t_length, n_data, n_obs, n_outliers):
    # TODO: what does this do? Write two lines.
    # Maybe do something as simple as possible (that still somehow checks the algorithm)?
    n_obs = int(n_obs)
    rand = np.random.RandomState(42)
    dt = t_length / n_data
    t = np.linspace(0, t_length, n_data)    
    t_obs = np.sort(rand.choice(t, n_obs, replace=False))
    n_outliers = n_outliers
    dmag = np.random.normal(0, 1, n_data) * -1**(rand.randint(2, size=n_data))
    dmag_obs = dmag[np.searchsorted(t, t_obs)]
    outliers = rand.randint(0, t.size, n_outliers)    
    mag = amplitude * np.sin(2 * np.pi * t / period) + dmag

    for n in range(n_outliers):
        mask = (t >= outliers[n])
        mag[mask] = mag[mask] + 10 * amplitude * np.exp(-1 * (t[mask] - outliers[n]))

    mag_obs = mag[np.searchsorted(t, t_obs)]
    return dict(t=t_obs, dt=dt, y=mag_obs, dy = dmag_obs)

TEST_CASES = [
    dict(period = 7, amplitude = 2, t_length = 100, n_data = 1000,
         n_observations = 1000 / 2, n_outliers = 0, dt = 0.01,
         sig_criterion = 'boot', significance = 95, n_bootstraps = 100),
    dict(period = 7, amplitude = 2, t_length = 100, n_data = 1000,
         n_observations = 1000 / 10, n_outliers = 0, dt = 0.01,
         sig_criterion = 'boot', significance = 95, n_bootstraps = 100),
    dict(period = 7, amplitude = 2, t_length = 100, n_data = 1000,
         n_observations = 1000 / 2, n_outliers = 0, dt = 0.01,
         sig_criterion = 'pre', significance = 95, n_bootstraps = 'None'),
]

@pytest.mark.parametrize('test_case', TEST_CASES)
def test_lomb_scargle(test_case):
    test_data = simulate_test_data(
        test_case['period'], test_case['amplitude'], test_case['t_length'],
        test_case['n_data'], test_case['n_observations'], test_case['n_outliers'],
    )
    result = lomb_scargle(
        test_data['t'], test_data['y'], test_data['dy'], test_case['dt'],
        test_case['sig_criterion'], test_case['n_bootstraps'],
    )
    assert_allclose(
        result['period'],
        test_case['period'],
        atol=test_case['dt'],
    )
    assert np.greater(
        result['significance'],
        test_case['significance'],
    )

@pytest.mark.parametrize('test_case', TEST_CASES)
def test_lomb_scargle_plot(test_case):
    test_data = simulate_test_data(
        test_case['period'], test_case['amplitude'], test_case['t_length'],
        test_case['n_data'], test_case['n_observations'], test_case['n_outliers']
    )
    result = lomb_scargle(
        test_data['t'], test_data['y'], test_data['dy'], test_case['dt'],
        test_case['sig_criterion'], test_case['n_bootstraps'],
    )
    lomb_scargle_plot(
        test_data['t'], test_data['y'], test_data['dy'], result['fgrid'],
        result['psd'], result['period'],
        result['significance'], result['swf']
    )

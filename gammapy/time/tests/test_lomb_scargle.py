import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..lomb_scargle import lomb_scargle


def simulate_test_data(period, amplitude, t_length, n_data, n_obs, n_outliers):
    """Function for creating an unevenly data biased by outliers.

    As underlying model, a single harmonic is used.
    First, an evenly sampled test data set is generated.
    It the is distorted by randomly choosing data points.
    Outliers are simulated as exponential burst with ten times the
    amplitude decreasing with a characteristic time length of 1.
    Flux errors are assumed to be gaussian and homoscedastic.

    It returns arrays for time, flux and flux error and the resolution of the test data.
    
    Parameters
    ----------
    period : `float`
        period of model
    amplitude : `float`
        amplitude of model
    t_length : `float`
        length of data set in units of time
    n_data : `float`
        number of points for the test data
    n_obs : `float`
        number of unevenly data points for the observation data
    n_outliers : `float`
        number of outliers in the test data

    Returns
    ----------
    t : `~numpy.ndarray`
        time for observation
    dt : `float`
        time resolution of test data
    y : `~numpy.ndarray`
        flux of observation
    dy : `~numpy.ndarray`
        flux error of observation
    """
    n_obs = int(n_obs)
    rand = np.random.RandomState(42)
    dt = t_length / n_data
    t = np.linspace(0, t_length, n_data)
    t_obs = np.sort(rand.choice(t, n_obs, replace=False))
    n_outliers = n_outliers
    dmag = np.random.normal(0, 1, n_data) * -1 ** (rand.randint(2, size=n_data))
    dmag_obs = dmag[np.searchsorted(t, t_obs)]
    outliers = rand.randint(0, t.size, n_outliers)
    mag = amplitude * np.sin(2 * np.pi * t / period) + dmag

    for n in range(n_outliers):
        mask = (t >= outliers[n])
        mag[mask] = mag[mask] + 10 * amplitude * np.exp(-1 * (t[mask] - outliers[n]))

    mag_obs = mag[np.searchsorted(t, t_obs)]
    return dict(t=t_obs, dt=dt, y=mag_obs, dy=dmag_obs)


@pytest.mark.parametrize('test_case', [
    dict(period=7, amplitude=2, t_length=100, n_data=1000,
         n_observations=1000 / 2, n_outliers=0, dt=0.01,
         max_period=10, criteria='boot', n_bootstraps=100),
    dict(period=7, amplitude=2, t_length=100, n_data=1000,
         n_observations=1000 / 2, n_outliers=0, dt=0.01,
         max_period='None', criteria='None', n_bootstraps='None'),
])
def test_lomb_scargle(test_case):
    test_data = simulate_test_data(
        test_case['period'], test_case['amplitude'], test_case['t_length'],
        test_case['n_data'], test_case['n_observations'], test_case['n_outliers'],
    )
    result = lomb_scargle(
        test_data['t'], test_data['y'], test_data['dy'], test_case['dt'],
        test_case['max_period'], test_case['criteria'], test_case['n_bootstraps'],
    )
    assert_allclose(result['period'], test_case['period'], atol=test_case['dt'], )

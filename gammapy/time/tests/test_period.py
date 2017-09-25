# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ...utils.testing import requires_dependency
from ..period import lomb_scargle


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
    n_data : `int`
        number of points for the test data
    n_obs : `int`
        number of unevenly data points for the observation data
    n_outliers : `int`
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
    rand = np.random.RandomState(42)
    dt = t_length / n_data
    t = np.linspace(0, t_length, n_data)
    t_obs = np.sort(rand.choice(t, n_obs, replace=False))
    n_outliers = n_outliers
    dmag = rand.normal(0, 1, n_data) * -1 ** (rand.randint(2, size=n_data))
    dmag_obs = dmag[np.searchsorted(t, t_obs)]
    outliers = rand.randint(0, t.size, n_outliers)
    mag = amplitude * np.sin(2 * np.pi * t / period) + dmag

    for n in range(n_outliers):
        mask = (t >= outliers[n])
        mag[mask] = mag[mask] + 10 * amplitude * np.exp(-1 * (t[mask] - outliers[n]))

    mag_obs = mag[np.searchsorted(t, t_obs)]
    return dict(t=t_obs, dt=dt, y=mag_obs, dy=dmag_obs)


@requires_dependency('scipy')
@pytest.mark.parametrize('pars', [
    dict(
        period=7, amplitude=2, t_length=100, n_data=1000,
        n_obs=500, n_outliers=0, dt=0.5,
        max_period=None, criteria='all', n_bootstraps=10,
        fap=dict(
            pre=2.220446049250313e-14,
            cvm=1.4011014570769476e-11,
            nll=5.6590954145008254e-09,
            boot=0,
        ),
    ),
])
def test_lomb_scargle(pars):
    test_data = simulate_test_data(
        pars['period'], pars['amplitude'], pars['t_length'],
        pars['n_data'], pars['n_obs'], pars['n_outliers'],
    )

    result = lomb_scargle(
        test_data['t'], test_data['y'], test_data['dy'], pars['dt'],
        pars['max_period'], pars['criteria'], pars['n_bootstraps'],
    )

    assert_allclose(result['period'], pars['period'], atol=pars['dt'])
    assert_allclose(result['fap']['pre'], pars['fap']['pre'], rtol=1e-3)
    assert_allclose(result['fap']['cvm'], pars['fap']['cvm'], rtol=1e-3)
    assert_allclose(result['fap']['nll'], pars['fap']['nll'], rtol=1e-3)
    assert_allclose(result['fap']['boot'], pars['fap']['boot'], rtol=1e-3)

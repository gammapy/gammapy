# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from collections import OrderedDict
from astropy.stats.lombscargle import _statistics
from ..period import robust_periodogram
from ...utils.testing import requires_dependency


def simulate_test_data(period, amplitude, t_length, n_data, n_obs, n_outliers):
    """
    Function for creating an unevenly data biased by outliers.

    As underlying model, a single harmonic model is chosen.
    First, an evenly sampled test data set is generated.
    It is distorted by randomly drawing data points from an uniform distribution.
    Outliers are simulated as exponential burst with ten times the
    amplitude decreasing with a characteristic time length of 1.
    Flux errors are assumed to be gaussian and homoscedastic.

    Returns arrays for time, flux and flux error and the resolution of the test data.

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


def fap_astropy(power, freq, t, y, dy, method=dict(baluev=0)):
    """
    Calls the `~astropy`-method `false_alarm_probability`.
    Function for estimation of periodogram peak significance.
    Assumes Gaussian white noise light curve.

    Returns an OrderedDict with the false alarm probability for each method.

    Parameters
    ----------
    power : `~numpy.ndarray`
        power of periodogram
    freq : `~numpy.ndarray`
        frequencies at withc the periodogram is computed
    t, y, dy : `~numpy.ndarray`
        time, flux and flux error or light curve
    method : `dict`
        dictionary of methods with their respective false alarm probability

    Returns
    -------
    fap : `~collections.OrderedDict`
        false alarm probability dictionary (see description above).
    """

    fap = OrderedDict()

    if 'single' in [*method]:
        fap['single'] = _statistics.false_alarm_probability(power.max(), freq.max(), t, y, dy, 'standard', 'single')
    if 'naive' in [*method]:
        fap['naive'] = _statistics.false_alarm_probability(power.max(), freq.max(), t, y, dy, 'standard', 'naive')
    if 'davies' in [*method]:
        fap['davies'] = _statistics.false_alarm_probability(power.max(), freq.max(), t, y, dy, 'standard', 'davies')
    if 'baluev' in [*method]:
        fap['baluev'] = _statistics.false_alarm_probability(power.max(), freq.max(), t, y, dy, 'standard', 'baluev')
    if 'bootstrap' in [*method]:
        fap['bootstrap'] = _statistics.false_alarm_probability(power.max(), freq.max(), t, y, dy, 'standard',
                                                               'bootstrap', dict(n_bootstraps=100, random_seed=42))

    return fap


@requires_dependency('scipy')
@pytest.mark.parametrize('pars', [
    dict(
        period=7, amplitude=2, t_length=100,
        n_data=1000, n_obs=500, n_outliers=50,
        periods=np.linspace(0.5, 100, 200), loss='cauchy', scale=1,
        fap=dict(
            single=2.855680054527823e-93,
            naive=5.705643031869405e-91,
            davies=6.752853065345455e-90,
            baluev=6.752853065345455e-90,
            bootstrap=0.43,
        ),
    ),
])
def test_period(pars):
    test_data = simulate_test_data(
        pars['period'], pars['amplitude'], pars['t_length'],
        pars['n_data'], pars['n_obs'], pars['n_outliers'],
    )

    periodogram = robust_periodogram(
        test_data['t'], test_data['y'], test_data['dy'],
        periods=pars['periods'], loss=pars['loss'], scale=pars['scale'],
    )

    fap = fap_astropy(periodogram['power'], 1. / periodogram['periods'],
                      test_data['t'], test_data['y'], test_data['dy'], pars['fap']
                      )

    assert_allclose(periodogram['best_period'], pars['period'], atol=pars['periods'].min())
    assert_allclose(fap['single'], pars['fap']['single'], rtol=1e-3)
    assert_allclose(fap['naive'], pars['fap']['naive'], rtol=1e-3)
    assert_allclose(fap['davies'], pars['fap']['davies'], rtol=1e-3)
    assert_allclose(fap['baluev'], pars['fap']['baluev'], rtol=1e-3)
    assert_allclose(fap['bootstrap'], pars['fap']['bootstrap'], rtol=1e-3)

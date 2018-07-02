# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np

__all__ = [
    'robust_periodogram',
]


def robust_periodogram(time, flux, flux_err=np.array([None]), periods=np.array([None]), loss='linear', scale=1):
    """
    Computes the period of a light curve by robust regression techniques assuming a single harmonic model.

    To compute the periodogram peaks with robust regression, the scipy object `~scipy.optimize.least_squares` is called.
    The false alarm probability of the highest periodogram peak can be computed with the `~false_alarm_probability`-method of `~astropy`.
    It assumes Gaussian white noise light curves.
    To evaluate the impact of the sampling pattern on the periodogram, it is recommend to compute the spectral window function with `~astropy`'s `astropy.stats.LombScargle`-class.

    For an introduction to the false alarm probability of periodogram peaks, see [1]_.
    For an introduction to robust regression techniques and loss functions provided by scipy, see [2]_ and [3]_.

    The function returns a results dictionary with the following content:

    - ``periods`` (`~numpy.ndarray`) -- Period grid in units of ``t``
    - ``power`` (`~numpy.ndarray`) -- Periodogram peaks at periods of ``pgrid``
    - ``best_period`` (`float`) -- Location of the highest periodogram peak

    Parameters
    ----------
    time : `~numpy.ndarray`
        Time array of the light curve
    flux : `~numpy.ndarray`
        Flux array of the light curve
    flux_err : `~numpy.ndarray` (optional, default=None)
        Flux error array of the light curve.
        Is set to 1 if not given.
    periods : `~numpy.ndarray` (optional, default described below)
        Period grid on which the periodogram is performed.
        If not given, a linear grid will be computed limited by the length of the light curve and the Nyquist frequency.
    loss : `str` (optional, default='linear')
        Loss function for the robust regression.
        Available: `{'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}`.
        If not given, `'linear'` will be used, resulting in the Lomb-Scargle periodogram.
    scale : `float` (optional, default=1)
        Loss scale parameter to define margin between inlier and outlier residuals.
        If not given, will be set to 1.

    Returns
    -------
    results : `~collections.OrderedDict`
        Results dictionary (see description above).

    References
    ----------
    .. [1] Astropy docs, `Link <http://docs.astropy.org/en/stable/stats/lombscargle.html>`_
    .. [2] Nikolay Mayorov (2015), "Robust nonlinear regression in scipy", `Link <http://scipy-cookbook.readthedocs.io/items/robust_regression.html>`_
    .. [3] Thieler et at. (2016), "RobPer: An R Package to Calculate Periodograms for Light Curves Based on Robust Regression",
       `Link <https://www.jstatsoft.org/article/view/v069i09>`_
    """
    # set flux errors
    if flux_err.any() == None:
        flux_err = np.ones_like(flux)

    # set up period grid
    if periods.any() == None:
        periods = _period_grid(time)

    # comnpute periodogram
    psd_data = _robust_regression(time, flux, flux_err, periods, loss, scale)

    # find period with highest periodogram peak
    best_period = periods[np.argmax(psd_data)]

    return OrderedDict([
        ('periods', periods),
        ('power', psd_data),
        ('best_period', best_period),
    ])


def _period_grid(time):
    """
    Generates the period grid for the periodogram
    """
    number_obs = len(time)
    length_lc = np.max(time) - np.min(time)

    dt = 2 * length_lc / number_obs
    max_period = np.rint(length_lc / dt) * dt
    min_period = dt

    periods = np.arange(min_period, max_period + dt, dt)

    return periods


def _model(beta0, x, period, t, y, dy):
    """
    Computes the residuals of the periodic model
    """
    x[:, 1] = np.cos(2 * np.pi * t / period)
    x[:, 2] = np.sin(2 * np.pi * t / period)

    return (y - np.dot(x, beta0.T)) / dy


def _noise(mu, t, y, dy):
    """
    Computes the residuals of the noise-only model
    """

    return (mu * np.ones(len(t)) - y) / dy


def _robust_regression(time, flux, flux_err, periods, loss, scale):
    """
    Computes the periodogram peaks for given loss function and scale
    """
    from scipy.optimize import least_squares

    beta0 = np.array([0, 1, 0])
    mu = np.median(flux)
    x = np.ones([len(time), len(beta0)])
    chi_model = np.empty([len(periods)])
    chi_noise = np.empty([len(periods)])

    for i in range(len(periods)):
        chi_model[i] = least_squares(_model, beta0, loss=loss, f_scale=scale,
                                     args=(x, periods[i], time, flux, flux_err)).cost
        chi_noise[i] = least_squares(_noise, mu, loss=loss, f_scale=scale,
                                     args=(time, flux, flux_err)).cost
    power = 1 - chi_model / chi_noise

    return power

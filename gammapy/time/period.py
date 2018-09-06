# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np

__all__ = ["robust_periodogram"]


def robust_periodogram(time, flux, flux_err=None, periods=None, loss="linear", scale=1):
    """
    Compute a light curve's period.

    A single harmonic model is fitted to the light curve.
    The periodogram returns the power for each period.
    The maximum power indicates the period of the light curve, assuming an underlying periodic process.

    The fitting can be done by ordinary least square regression (Lomb-Scargle periodogram) or robust regression.
    For robust regression, the scipy object `~scipy.optimize.least_squares` is called.
    For an introduction to robust regression techniques and loss functions, see [1]_ and [2]_.

    The significance of a periodogram peak can be evaluated in terms of a false alarm probability.
    It can be computed with the `~false_alarm_probability`-method of `~astropy`, assuming Gaussian white noise light curves.
    For an introduction to the false alarm probability of periodogram peaks, see :ref:`stats-lombscargle`.

    The periodogram is biased by measurement errors, high order modes and sampling of the light curve.
    To evaluate the impact of the sampling, compute the spectral window function with the `astropy.stats.LombScargle`-class.

    The function returns a dictionary with the following content:

    - ``periods`` (`numpy.ndarray`) -- Period grid in units of ``t``
    - ``power`` (`numpy.ndarray`) -- Periodogram peaks at periods of ``pgrid``
    - ``best_period`` (float) -- Period of the highest periodogram peak

    Parameters
    ----------
    time : `numpy.ndarray`
        Time array of the light curve
    flux : `numpy.ndarray`
        Flux array of the light curve
    flux_err : `numpy.ndarray`
        Flux error array of the light curve. Default is 1.
    periods : `numpy.ndarray`
        Period grid on which the periodogram is performed.
        If not given, a linear grid will be computed limited by the length of the light curve and the Nyquist frequency.
    loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}
        Loss function for the robust regression.
        Default is 'linear', resulting in the Lomb-Scargle periodogram.
    scale : float (optional, default=1)
        Loss scale parameter to define margin between inlier and outlier residuals.
        If not given, will be set to 1.

    Returns
    -------
    results : `~collections.OrderedDict`
        Results dictionary (see description above).

    References
    ----------
    .. [1] Nikolay Mayorov (2015), "Robust nonlinear regression in scipy",
       see `here <http://scipy-cookbook.readthedocs.io/items/robust_regression.html>`__
    .. [2] Thieler et at. (2016), "RobPer: An R Package to Calculate Periodograms for Light Curves Based on Robust Regression",
       see `here <https://www.jstatsoft.org/article/view/v069i09>`__
    """
    if flux_err is None:
        flux_err = np.ones_like(flux)

    # set up period grid
    if periods is None:
        periods = _period_grid(time)

    # compute periodogram
    psd_data = _robust_regression(time, flux, flux_err, periods, loss, scale)

    # find period of highest periodogram peak
    best_period = periods[np.argmax(psd_data)]

    return OrderedDict(
        [("periods", periods), ("power", psd_data), ("best_period", best_period)]
    )


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
    Residuals of the noise-only model.
    """
    return (mu * np.ones(len(t)) - y) / dy


def _robust_regression(time, flux, flux_err, periods, loss, scale):
    """
    Periodogram peaks for a given loss function and scale.
    """
    from scipy.optimize import least_squares

    beta0 = np.array([0, 1, 0])
    mu = np.median(flux)
    x = np.ones([len(time), len(beta0)])
    chi_model = np.empty([len(periods)])
    chi_noise = np.empty([len(periods)])

    for i in range(len(periods)):
        chi_model[i] = least_squares(
            _model,
            beta0,
            loss=loss,
            f_scale=scale,
            args=(x, periods[i], time, flux, flux_err),
        ).cost
        chi_noise[i] = least_squares(
            _noise, mu, loss=loss, f_scale=scale, args=(time, flux, flux_err)
        ).cost
    power = 1 - chi_model / chi_noise

    return power

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.optimize

__all__ = ["robust_periodogram", "plot_periodogram"]


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
    To evaluate the impact of the sampling, compute the spectral window function with the `astropy.timeseries.LombScargle` class.

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
    results : `dict`
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

    return {"periods": periods, "power": psd_data, "best_period": best_period}


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
    beta0 = np.array([0, 1, 0])
    mu = np.median(flux)
    x = np.ones([len(time), len(beta0)])
    chi_model = np.empty([len(periods)])
    chi_noise = np.empty([len(periods)])

    for i in range(len(periods)):
        chi_model[i] = scipy.optimize.least_squares(
            _model,
            beta0,
            loss=loss,
            f_scale=scale,
            args=(x, periods[i], time, flux, flux_err),
        ).cost
        chi_noise[i] = scipy.optimize.least_squares(
            _noise, mu, loss=loss, f_scale=scale, args=(time, flux, flux_err)
        ).cost
    power = 1 - chi_model / chi_noise

    return power


def plot_periodogram(
    time, flux, periods, power, flux_err=None, best_period=None, fap=None
):
    """
    Plot a light curve and its periodogram.

    The highest period of the periodogram and its false alarm probability (FAP) is added to the plot, if given.

    Parameters
    ----------
    time : `numpy.ndarray`
        Time array of the light curve
    flux : `numpy.ndarray`
        Flux array of the light curve
    periods : `numpy.ndarray`
        Periods for the periodogram
    power : `numpy.ndarray`
        Periodogram peaks of the data
    flux_err : `numpy.ndarray` (optional, default=None)
        Flux error array of the light curve.
        Is set to 0 if not given.
    best_period : float (optional, default=None)
        Period of the highest periodogram peak
    fap : float (optional, default=None)
        False alarm probability of ``best_period`` under a certain significance criterion.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if flux_err is None:
        flux_err = np.zeros_like(flux)

    # set up the figure & axes for plotting
    fig = plt.figure(figsize=(16, 9))
    grid_spec = plt.GridSpec(2, 1)

    # plot the light curve
    ax = fig.add_subplot(grid_spec[0, :])
    ax.errorbar(
        time, flux, flux_err, fmt="ok", label="light curve", elinewidth=1.5, capsize=0
    )
    ax.set_xlabel("time")
    ax.set_ylabel("flux")
    ax.legend()

    # plot the periodogram
    ax = fig.add_subplot(grid_spec[1, :])
    ax.plot(periods, power, c="k", label="periodogram")
    # mark the best period and label with significance
    if best_period is not None:
        if fap is None:
            raise ValueError(
                "Must give a false alarm probability if you give a best_period"
            )

        # set precision for period format
        pre = int(abs(np.floor(np.log10(np.max(np.diff(periods))))))
        label = "Detected period p = {:.{}f} with {:.2E} FAP".format(
            best_period, pre, fap
        )
        ymax = power[periods == best_period]
        ax.axvline(best_period, ymin=0, ymax=ymax, label=label, c="r")

    ax.set_xlabel("period")
    ax.set_ylabel("power")
    ax.set_xlim(0, np.max(periods))
    ax.legend()

    return fig

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.stats as stats
import astropy.units as u

__all__ = [
    "compute_fvar",
    "compute_fpp",
    "compute_dtime",
    "compute_chisq",
]


def compute_fvar(flux, flux_err, axis=0):
    r"""Calculate the fractional excess variance.

    This method accesses the ``FLUX`` and ``FLUX_ERR`` columns
    from the lightcurve data.

    The fractional excess variance :math:`F_{var}`, an intrinsic
    variability estimator, is given by

    .. math::
        F_{var} = \sqrt{ \frac{S^{2} - \bar{ \sigma^{2}}}{ \bar{x}^{2}}}

    It is the excess variance after accounting for the measurement errors
    on the light curve :math:`\sigma`. :math:`S` is the variance.

    It is important to note that the errors on the flux must be gaussian.

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        the measured fluxes
    flux_err : `~astropy.units.Quantity`
        the error on measured fluxes
    axis : int, optional
        Axis along which the excess variance is computed.
        The default is to compute the value on axis 0.

    Returns
    -------
    fvar, fvar_err : `~numpy.ndarray`
        Fractional excess variance.

    References
    ----------
    .. [Vaughan2003] "On characterizing the variability properties of X-ray light
       curves from active galaxies", Vaughan et al. (2003)
       https://ui.adsabs.harvard.edu/abs/2003MNRAS.345.1271V
    """

    flux_mean = np.nanmean(flux, axis=axis)
    n_points = np.count_nonzero(~np.isnan(flux), axis=axis)

    s_square = np.nansum((flux - flux_mean) ** 2, axis=axis) / (n_points - 1)
    sig_square = np.nansum(flux_err**2, axis=axis) / n_points
    fvar = np.sqrt(np.abs(s_square - sig_square)) / flux_mean

    sigxserr_a = np.sqrt(2 / n_points) * sig_square / flux_mean**2
    sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fvar / flux_mean)
    sigxserr = np.sqrt(sigxserr_a**2 + sigxserr_b**2)
    fvar_err = sigxserr / (2 * fvar)

    return fvar, fvar_err


def compute_fpp(flux, flux_err, axis=0):
    r"""Calculate the point-to-point excess variance.

    This method accesses the ``FLUX`` and ``FLUX_ERR`` columns
    from the lightcurve data.

    F_pp is a quantity strongly related to F_var, probing the variability
    in a shorter timescale.

    For white noise, F_pp and F_var give the same value.
    However, for red noise, F_var will be larger
    than F_pp, as the variations will be larger on longer timescales

    It is important to note that the errors on the flux must be gaussian.

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        the measured fluxes
    flux_err : `~astropy.units.Quantity`
        the error on measured fluxes
    axis : int, optional
        Axis along which the excess variance is computed.
        The default is to compute the value on axis 0.

    Returns
    -------
    fpp, fpp_err : `~numpy.ndarray`
        Point-to-point excess variance.

    References
    ----------
    .. [Edelson2002] "X-Ray Spectral Variability and Rapid Variability
       of the Soft X-Ray Spectrum Seyfert 1 Galaxies
       Arakelian 564 and Ton S180", Edelson et al. (2002)
       https://iopscience.iop.org/article/10.1086/323779
    """

    flux_mean = np.nanmean(flux, axis=axis)
    n_points = np.count_nonzero(~np.isnan(flux), axis=axis)
    flux = np.atleast_2d(flux).swapaxes(0, axis).T

    s_square = np.nansum((flux[..., 1:] - flux[..., :-1]) ** 2, axis=-1) / (
        n_points.T - 1
    )
    sig_square = np.nansum(flux_err**2, axis=axis) / n_points
    fpp = np.sqrt(np.abs(s_square.T - sig_square)) / flux_mean

    sigxserr_a = np.sqrt(2 / n_points) * sig_square / flux_mean**2
    sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fpp / flux_mean)
    sigxserr = np.sqrt(sigxserr_a**2 + sigxserr_b**2)
    fpp_err = sigxserr / (2 * fpp)

    return fpp, fpp_err


def compute_chisq(flux):
    r"""Calculate the chi-square test for `LightCurve`.

    Chisquare test is a variability estimator. It computes
    deviations from the expected value here mean value

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        the measured fluxes

    Returns
    -------
    ChiSq, P-value : tuple of float or `~numpy.ndarray`
        Tuple of Chi-square and P-value
    """
    yexp = np.mean(flux)
    yobs = flux.data
    chi2, pval = stats.chisquare(yobs, yexp)
    return chi2, pval


def compute_dtime(flux, flux_err, time, axis=0):
    r"""Calculate the characteristic doubling time for a series of measurements.

    The characteristic doubling time  is estimated to obtain the
    minimum variability timescale for the light curves in which
    rapid variations are clearly evident: for example it is useful in AGN flaring episodes.

    This quantity, especially for AGN flares, is often expressed
    as the pair of doubling time and halving time, or the minimum characteristic time
    for the rising and falling components respectively.

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        the measured fluxes
    flux_err : `~astropy.units.Quantity`
        the error on measured fluxes
    time: `~astropy.units.Quantity`
        the times at which the fluxes are measured
    axis : int, optional
        Axis along which the value is computed.
        The default is to compute the value on axis 0.

    Returns
    -------
    dtime, dtime_err : `~numpy.ndarray`
        Characteristic doubling time, halving time and errors.

    References
    ----------
    ..[Brown2013] "Locating the γ-ray emission region
    of the flat spectrum radio quasar PKS 1510−089", Brown et al. (2013)
    https://academic.oup.com/mnras/article/431/1/824/1054498
    """

    flux = np.atleast_2d(flux).swapaxes(0, axis).T
    flux_err = np.atleast_2d(flux_err).swapaxes(0, axis).T

    times = (time[1:] - time[:-1]) / np.log2(flux[..., 1:] / flux[..., :-1])
    times_err_1 = (
        (time[1:] - time[:-1])
        * np.log(2)
        / flux[..., 1:]
        * np.log(flux[..., 1:] / flux[..., :-1]) ** 2
    )
    times_err_2 = (
        (time[1:] - time[:-1])
        * np.log(2)
        / flux[..., :-1]
        * np.log(flux[..., 1:] / flux[..., :-1]) ** 2
    )
    times_err = np.sqrt(
        (flux_err[..., 1:] * times_err_1) ** 2 + (flux_err[..., :-1] * times_err_2) ** 2
    )

    imin = np.argmin(
        np.where(np.logical_and(np.isfinite(times), times > 0), times, 1e15 * u.s),
        axis=-1,
        keepdims=True,
    )
    imax = np.argmax(
        np.where(np.logical_and(np.isfinite(times), times < 0), times, -1e15 * u.s),
        axis=-1,
        keepdims=True,
    )
    index = np.concatenate([imin, imax], axis=-1)

    dtime = np.take_along_axis(times, index, axis=-1)
    dtime_err = np.take_along_axis(times_err, index, axis=-1)

    return dtime, dtime_err

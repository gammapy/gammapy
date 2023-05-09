# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.stats as stats

__all__ = [
    "compute_fvar",
    "weighted_fvar",
    "compute_fpp",
    "compute_2time",
    "compute_etime",
    "compute_chisq",
    "lc_fvar",
    "lc_fpp",
    "eval_lc_timing",
]


def compute_fvar(flux, flux_err):
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
    If temporal bins are non-uniform in size, the simple fractional excess variance is

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        the measured fluxes
    flux_err : `~astropy.units.Quantity`
        the error on measured fluxes

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

    flux_mean = np.mean(flux)
    n_points = len(flux)

    s_square = np.sum((flux - flux_mean) ** 2) / (n_points - 1)
    sig_square = np.nansum(flux_err**2) / n_points
    fvar = np.sqrt(np.abs(s_square - sig_square)) / flux_mean

    sigxserr_a = np.sqrt(2 / n_points) * sig_square / flux_mean**2
    sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fvar / flux_mean)
    sigxserr = np.sqrt(sigxserr_a**2 + sigxserr_b**2)
    fvar_err = sigxserr / (2 * fvar)

    return fvar, fvar_err


def compute_fpp(flux, flux_err):
    r"""Calculate the point-to-point fractional variation.

    This method accesses the ``FLUX`` and ``FLUX_ERR`` columns
    from the lightcurve data.

    The point-to-point fractional variation F_pp probes variability on a shorter timescale
    than F_var.

    .. math::
        F_{pp} = \sqrt{\frac{\frac{\sum{(X_{i+1}-X_i)^2}}{2(N - 1)} - \bar{\sigma^{2}}}{\bar{x}^{2}}}.

    For white noise, F_var and F_pp give the same value. For red noise, F_var > F_pp

    It is important to note that the errors on the flux lust be gaussian.

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        the measured fluxes
    flux_err : `~astropy.units.Quantity`
        the error on measured fluxes

    Returns
    -------
    fpp : `~numpy.ndarray`
        Point-to-point fractional variation

    References
    ----------
    ..  [Edelson2002] "X-ray Spectral Variability and Rapid Variability of the Soft X-ray Spectrum
        Seyfert 1 Galaxies Akn 564 and Ton S180", Edelson et al. (2002)
        https://ui.adsabs.harvard.edu/abs/2002ApJ...568..610E/abstract
    """

    flux_mean = np.mean(flux)
    n_points = len(flux)

    s_square = np.sum((flux[1:] - flux[:-1]) ** 2) / (2 * (n_points - 1))
    sig_square = np.nansum(flux_err**2) / n_points
    fpp = np.sqrt(np.abs(s_square - sig_square)) / flux_mean

    return fpp


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


def compute_2time(flux, time):
    r"""Calculate the doubling/halving time for `LightCurve`.

    The doubling or halving time is estimated to obtain
    the minimum variability timescale for the light curves
    in which rapid variations are clearly evident during the flaring episodes.

    It is obtained as :math: \tau = min(T_2 ^{ij}) where

    ..math::
          T_2 ^{ij} = \frac{F_i + F_j}{2} \lvert \frac{t_j - t_i}{F_j - F_i} \rvert

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        the measured fluxes
    time : `~astropy.Time`
        the time bin centers

    Returns
    -------
    t2 : float
        the doubling/halving time

    References
    ----------
    [Zhang1999]Rapid X-Ray Variability of the BL Lacertae Object PKS 2155–304,
    Zhang et al. 1999
    https://iopscience.iop.org/article/10.1086/308116/meta
    """
    tij = []
    for i in range(len(flux) - 1):
        for j in range(i + 1, len(flux)):
            t = 0.5 * (flux[i] + flux[j]) * ((time[j] - time[i]) / (flux[j] - flux[i]))
            tij.append(t)

    t2 = np.amin(tij)

    return t2


def compute_etime(flux, time):
    r"""Calculate the e-folding time for `LightCurve`.

    The e-folding time is defined as the characteristic time required
    for the flux to change by a factor ::math:: e^\pm

    It is useful int he case of exponential evolution of a flare
    It is obtained as :math: \tau = min(T_e ^{ij}) where
    ..math::
          T_e ^{ij} = \lvert \frac{t_j - t_i}{\ln{F_j} - \ln{F_i}} \rvert

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        the measured fluxes
    time : `~astropy.Time`
        the time bin centers

    Returns
    -------
    te : float
        the e-folding time

    References
    ----------
    [Calderone2011]Gamma-ray variability of radio-loud narrow-line Seyfert 1 galaxies,
    Calderone et al. 2011
    https://academic.oup.com/mnras/article/413/4/2365/962486
    """
    tij = []
    for i in range(len(flux) - 1):
        for j in range(i + 1, len(flux)):
            t = (time[j] - time[i]) / (np.log(flux[i]) - np.log(flux[j]))
            tij.append(t)

    te = np.amin(tij)

    return te
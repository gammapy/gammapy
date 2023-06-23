# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.stats as stats

__all__ = ["compute_fvar", "compute_chisq"]


def compute_fvar(flux, flux_err):
    r"""Calculate the fractional excess variance.

    This method accesses the the ``FLUX`` and ``FLUX_ERR`` columns
    from the lightcurve data.

    The fractional excess variance :math:`F_{var}`, an intrinsic
    variability estimator, is given by

    .. math::
        F_{var} = \sqrt{\frac{S^{2} - \bar{\sigma^{2}}}{\bar{x}^{2}}}.

    It is the excess variance after accounting for the measurement errors
    on the light curve :math:`\sigma`. :math:`S` is the variance.

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


def compute_chisq(flux):
    """Calculate the chi-square test for `LightCurve`.

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

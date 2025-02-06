# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Common fit statistics used in gamma-ray astronomy.

see :ref:`fit-statistics`
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.special import erfc
from gammapy.stats.fit_statistics_cython import TRUNCATION_VALUE, cash_sum_cython

__all__ = [
    "cash",
    "cstat",
    "wstat",
    "get_wstat_mu_bkg",
    "get_wstat_gof_terms",
    "CashFitStatistic",
    "WStatFitStatistic",
    "Chi2FitStatistic",
    "Chi2AsymmetricErrorFitStatistic",
    "FIT_STATISTICS_REGISTRY",
]


def cash(n_on, mu_on, truncation_value=TRUNCATION_VALUE):
    r"""Cash statistic, for Poisson data.

    The Cash statistic is defined as:

    .. math::
        C = 2 \left( \mu_{on} - n_{on} \log \mu_{on} \right)

    and :math:`C = 0` where :math:`\mu <= 0`.
    For more information see :ref:`fit-statistics`.

    Parameters
    ----------
    n_on : `~numpy.ndarray` or array_like
        Observed counts.
    mu_on : `~numpy.ndarray` or array_like
        Expected counts.
    truncation_value : `~numpy.ndarray` or array_like
        Minimum value use for ``mu_on``
        ``mu_on`` = ``truncation_value`` where ``mu_on`` <= ``truncation_value``.
        Default is 1e-25.

    Returns
    -------
    stat : ndarray
        Statistic per bin.

    References
    ----------
    * `Sherpa statistics page section on the Cash statistic:
      <http://cxc.cfa.harvard.edu/sherpa/statistics/#cash>`_
    * `Sherpa help page on the Cash statistic:
      <http://cxc.harvard.edu/sherpa/ahelp/cash.html>`_
    * `Cash 1979, ApJ 228, 939,
      <https://ui.adsabs.harvard.edu/abs/1979ApJ...228..939C>`_
    """
    n_on = np.asanyarray(n_on)
    mu_on = np.asanyarray(mu_on)
    truncation_value = np.asanyarray(truncation_value)
    if np.any(truncation_value) <= 0:
        raise ValueError("Cash statistic truncation value must be positive.")

    mu_on = np.where(mu_on <= truncation_value, truncation_value, mu_on)

    # suppress zero division warnings, they are corrected below
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = 2 * (mu_on - n_on * np.log(mu_on))
    return stat


def cstat(n_on, mu_on, truncation_value=TRUNCATION_VALUE):
    r"""C statistic, for Poisson data.

    The C statistic is defined as:

    .. math::
        C = 2 \left[ \mu_{on} - n_{on} + n_{on}
            (\log(n_{on}) - log(\mu_{on}) \right]

    and :math:`C = 0` where :math:`\mu_{on} <= 0`.

    ``truncation_value`` handles the case where ``n_on`` or ``mu_on`` is 0 or less and
    the log cannot be taken.
    For more information see :ref:`fit-statistics`.

    Parameters
    ----------
    n_on : `~numpy.ndarray` or array_like
        Observed counts.
    mu_on : `~numpy.ndarray` or array_like
        Expected counts.
    truncation_value : float
        ``n_on`` = ``truncation_value`` where ``n_on`` <= ``truncation_value.``
        ``mu_on`` = ``truncation_value`` where ``n_on`` <= ``truncation_value``
        Default is 1e-25.

    Returns
    -------
    stat : ndarray
        Statistic per bin.

    References
    ----------
    * `Sherpa stats page section on the C statistic:
      <http://cxc.cfa.harvard.edu/sherpa/statistics/#cstat>`_
    * `Sherpa help page on the C statistic:
      <http://cxc.harvard.edu/sherpa/ahelp/cash.html>`_
    * `Cash 1979, ApJ 228, 939,
      <https://ui.adsabs.harvard.edu/abs/1979ApJ...228..939C>`_
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    mu_on = np.asanyarray(mu_on, dtype=np.float64)
    truncation_value = np.asanyarray(truncation_value, dtype=np.float64)

    if np.any(truncation_value) <= 0:
        raise ValueError("Cstat statistic truncation value must be positive.")

    n_on = np.where(n_on <= truncation_value, truncation_value, n_on)
    mu_on = np.where(mu_on <= truncation_value, truncation_value, mu_on)

    term1 = np.log(n_on) - np.log(mu_on)
    stat = 2 * (mu_on - n_on + n_on * term1)
    stat = np.where(mu_on > 0, stat, 0)

    return stat


def wstat(n_on, n_off, alpha, mu_sig, mu_bkg=None, extra_terms=True):
    r"""W statistic, for Poisson data with Poisson background.

    For a definition of WStat see :ref:`wstat`. If ``mu_bkg`` is not provided
    it will be calculated according to the profile likelihood formula.

    Parameters
    ----------
    n_on : `~numpy.ndarray` or array_like
        Total observed counts.
    n_off : `~numpy.ndarray` or array_like
        Total observed background counts.
    alpha : `~numpy.ndarray` or array_like
        Exposure ratio between on and off region.
    mu_sig : `~numpy.ndarray` or array_like
        Signal expected counts.
    mu_bkg : `~numpy.ndarray` or array_like, optional
        Background expected counts.
    extra_terms : bool, optional
        Add model independent terms to convert stat into goodness-of-fit
        parameter. Default is True.

    Returns
    -------
    stat : ndarray
        Statistic per bin.

    References
    ----------
    * `Habilitation M. de Naurois, p. 141,
      <http://inspirehep.net/record/1122589/files/these_short.pdf>`_
    * `XSPEC page on Poisson data with Poisson background,
      <https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html>`_
    """
    # Note: This is equivalent to what's defined on the XSPEC page under the
    # following assumptions
    # t_s * m_i = mu_sig
    # t_b * m_b = mu_bkg
    # t_s / t_b = alpha

    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)
    mu_sig = np.asanyarray(mu_sig, dtype=np.float64)

    if mu_bkg is None:
        mu_bkg = get_wstat_mu_bkg(n_on, n_off, alpha, mu_sig)

    term1 = mu_sig + (1 + alpha) * mu_bkg

    # suppress zero division warnings, they are corrected below
    with np.errstate(divide="ignore", invalid="ignore"):
        # This is a false positive error from pylint
        # See https://github.com/PyCQA/pylint/issues/2436
        term2_ = -n_on * np.log(mu_sig + alpha * mu_bkg)  # pylint:disable=invalid-unary-operand-type
    # Handle n_on == 0
    condition = n_on == 0
    term2 = np.where(condition, 0, term2_)

    # suppress zero division warnings, they are corrected below
    with np.errstate(divide="ignore", invalid="ignore"):
        # This is a false positive error from pylint
        # See https://github.com/PyCQA/pylint/issues/2436
        term3_ = -n_off * np.log(mu_bkg)  # pylint:disable=invalid-unary-operand-type
    # Handle n_off == 0
    condition = n_off == 0
    term3 = np.where(condition, 0, term3_)

    stat = 2 * (term1 + term2 + term3)

    if extra_terms:
        stat += get_wstat_gof_terms(n_on, n_off)

    return stat


def get_wstat_mu_bkg(n_on, n_off, alpha, mu_sig):
    """Background estimate ``mu_bkg`` for WSTAT.

    See :ref:`wstat`.
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)
    mu_sig = np.asanyarray(mu_sig, dtype=np.float64)

    # NOTE: Corner cases in the docs are all handled correctly by this formula
    C = alpha * (n_on + n_off) - (1 + alpha) * mu_sig
    D = np.sqrt(C**2 + 4 * alpha * (alpha + 1) * n_off * mu_sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_bkg = (C + D) / (2 * alpha * (alpha + 1))

    return mu_bkg


def get_wstat_gof_terms(n_on, n_off):
    """Goodness of fit terms for WSTAT.

    See :ref:`wstat`.
    """
    term = np.zeros(n_on.shape)

    # suppress zero division warnings, they are corrected below
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = -n_on * (1 - np.log(n_on))
        term2 = -n_off * (1 - np.log(n_off))

    term += np.where(n_on == 0, 0, term1)
    term += np.where(n_off == 0, 0, term2)

    return 2 * term


class FitStatistic(ABC):
    """Abstract base class for FitStatistic objects."""

    @staticmethod
    @abstractmethod
    def required_inputs():
        """Define the required argument names for this statistic."""
        pass

    @classmethod
    def stat_sum(cls, *args, **kwargs):
        """Calculate -2 * sum log(L)."""
        #        pass
        return np.sum(cls.stat_array(*args, **kwargs))

    @classmethod
    #    @abstractmethod
    def stat_array(cls, *args, **kwargs):
        """Calculate -2 * log(L)."""
        raise NotImplementedError

    @classmethod
    def loglikelihood(cls, *args, **kwargs):
        """Calculate sum log(L)."""
        return -0.5 * cls.stat_sum(*args, **kwargs)


class CashFitStatistic(FitStatistic):
    """Cash statistic class for Poisson with known background."""

    @staticmethod
    def required_inputs():
        return ["counts", "npred"]

    @staticmethod
    def stat_sum(counts, npred):
        counts = counts.astype(float)  # This might be done in the Dataset
        return cash_sum_cython(counts.ravel(), npred.ravel())

    @staticmethod
    def stat_array(counts, npred):
        """Statistic function value per bin given the current model parameters."""
        return cash(n_on=counts, mu_on=npred)


class WStatFitStatistic(FitStatistic):
    """WStat fit statistic class for ON-OFF Poisson measurements."""

    @staticmethod
    def required_inputs():
        return ["counts", "counts_off", "alpha", "npred_signal"]

    @staticmethod
    def stat_array(counts, counts_off, alpha, npred_signal):
        """Statistic function value per bin given the current model parameters."""
        on_stat_ = wstat(
            n_on=counts,
            n_off=counts_off,
            alpha=alpha,
            mu_sig=npred_signal,
        )
        return np.nan_to_num(on_stat_)


class Chi2FitStatistic(FitStatistic):
    """Chi2 fit statistic class for measurements with gaussian symmetric errors."""

    @staticmethod
    def required_inputs():
        return ["dnde", "flux_pred", "dnde_err"]

    @staticmethod
    def stat_array(data, model, sigma):
        """Statistic function value per bin given the current model."""
        return ((data - model) / sigma).to_value("") ** 2


class Chi2AsymmetricErrorFitStatistic(FitStatistic):
    """Pseudo-Chi2 fit statistic class for measurements with gaussian asymmetric errors with upper limits.

    Assumes that regular data follow asymmetric normal pdf and upper limits follow complementary error functions
    """

    @staticmethod
    def required_inputs():
        return ["dnde", "flux_pred", "dnde_errn", "dnde_errp"]

    @staticmethod
    def stat_array(data, model, errn, errp, is_ul=None, ul=None):
        """Asymmetric chi2 for asymmetric errors and UL.

        NaNs should be removed before calling in the function. All arrays should have the same size.

        Parameters
        ----------
        data : `~numpy.ndarray`
            the data array
        model : `~numpy.ndarray`
            the model array
        errn : `~numpy.ndarray`
            the negative error array
        errp : `~numpy.ndarray`
            the positive error array
        is_ul : `~numpy.ndarray`, optional
            the upper limit mask array. Mandatory if ul is passed.
        ul : `~numpy.ndarray`, optional
            the upper limit array. Mandatory if is_ul is passed.

        Returns
        -------
        stat_array : `~numpy.ndarray`
            the statistic array .
        """
        stat = np.zeros(model.shape)
        scale = np.zeros(model.shape)

        mask_p = model >= data
        scale[mask_p] = errp[mask_p]
        scale[~mask_p] = errn[~mask_p]

        stat = ((data - model) / scale) ** 2

        if is_ul is not None and ul is not None:
            value = model[is_ul]
            loc_ul = data[is_ul]
            scale_ul = ul[is_ul]
            stat[is_ul] = 2 * np.log(
                (erfc((loc_ul - value) / scale_ul) / 2)
                / (erfc((loc_ul - 0) / scale_ul) / 2)
            )

        return np.nan_to_num(stat)


FIT_STATISTICS_REGISTRY = {
    "cash": CashFitStatistic,
    "wstat": WStatFitStatistic,
    "chi2": Chi2FitStatistic,
    "distrib": Chi2AsymmetricErrorFitStatistic,
}

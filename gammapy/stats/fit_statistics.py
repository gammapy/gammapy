# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Common fit statistics used in gamma-ray astronomy.

see :ref:`fit-statistics`
"""

from abc import ABC
import numpy as np
from scipy.special import erfc
from gammapy.maps import Map
from gammapy.stats.fit_statistics_cython import (
    TRUNCATION_VALUE,
    cash_sum_cython,
    weighted_cash_sum_cython,
)


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
    * `Sherpa statistics page section on the Cash statistic
      <http://cxc.cfa.harvard.edu/sherpa/statistics/#cash>`_
    * `Sherpa help page on the Cash statistic
      <http://cxc.harvard.edu/sherpa/ahelp/cash.html>`_
    * `Cash (1979), ApJ 228, 939,
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
    * `Sherpa stats page section on the C statistic
      <http://cxc.cfa.harvard.edu/sherpa/statistics/#cstat>`_
    * `Sherpa help page on the C statistic
      <http://cxc.harvard.edu/sherpa/ahelp/cash.html>`_
    * `Cash (1979), ApJ 228, 939
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
    * `Habilitation M. de Naurois, p. 141
      <http://inspirehep.net/record/1122589/files/these_short.pdf>`_
    * `XSPEC page on Poisson data with Poisson background
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

    @classmethod
    def stat_sum_dataset(cls, dataset):
        """Calculate -2 * sum log(L)."""
        stat_array = cls.stat_array_dataset(dataset)
        if dataset.mask is not None:
            mask = dataset.mask.data if isinstance(dataset.mask, Map) else dataset.mask
            stat_array = stat_array[mask]
        return np.sum(stat_array)

    @classmethod
    def stat_array_dataset(cls, dataset):
        """Calculate -2 * log(L)."""
        raise NotImplementedError

    @classmethod
    def loglikelihood_dataset(cls, dataset):
        """Calculate sum log(L)."""
        return -0.5 * cls.stat_sum_dataset(dataset)


class CashFitStatistic(FitStatistic):
    """Cash statistic class for Poisson with known background."""

    @classmethod
    def stat_sum_dataset(cls, dataset):
        mask = dataset.mask
        counts, npred = dataset.counts.data, dataset.npred().data

        if mask is not None:
            mask = mask.data.astype("bool")
            counts, npred = counts[mask], npred[mask]

        counts = counts.astype(float)  # This might be done in the Dataset
        return cash_sum_cython(counts.ravel(), npred.ravel())

    @classmethod
    def stat_array_dataset(cls, dataset):
        counts, npred = dataset.counts.data, dataset.npred().data
        return cash(n_on=counts, mu_on=npred)


class WeightedCashFitStatistic(FitStatistic):
    """Cash statistic class for Poisson with known background applying weights."""

    @classmethod
    def stat_sum_dataset(cls, dataset):
        counts, npred = dataset.counts.data.astype(float), dataset.npred().data

        if dataset.mask is not None:
            mask = ~(dataset.mask.data == False)  # noqa
            counts = counts[mask]
            npred = npred[mask]

            weights = dataset.mask.data[mask].astype("float")
            return weighted_cash_sum_cython(counts, npred, weights)
        else:
            # No weights back to regular cash statistic
            return cash_sum_cython(counts.ravel(), npred.ravel())

    @classmethod
    def stat_array_dataset(cls, dataset):
        counts, npred = dataset.counts.data, dataset.npred().data
        weights = 1.0
        if dataset.mask is not None:
            weights = dataset.mask.astype("float")
        return cash(n_on=counts, mu_on=npred) * weights


class WStatFitStatistic(FitStatistic):
    """WStat fit statistic class for ON-OFF Poisson measurements."""

    @classmethod
    def stat_array_dataset(cls, dataset):
        """Statistic function value per bin given the current model parameters."""
        counts, counts_off, alpha = (
            dataset.counts.data,
            dataset.counts_off.data,
            dataset.alpha.data,
        )
        npred_signal = dataset.npred_signal().data
        on_stat_ = wstat(
            n_on=counts,
            n_off=counts_off,
            alpha=alpha,
            mu_sig=npred_signal,
        )
        return np.nan_to_num(on_stat_)

    @classmethod
    def stat_sum_dataset(cls, dataset):
        """Statistic function value per bin given the current model parameters."""
        if dataset.counts_off is None and not np.any(dataset.mask_safe.data):
            return 0
        else:
            stat_array = cls.stat_array_dataset(dataset)
            if dataset.mask is not None:
                stat_array = stat_array[dataset.mask.data]
            return np.sum(stat_array)


class Chi2FitStatistic(FitStatistic):
    """Chi2 fit statistic class for measurements with gaussian symmetric errors."""

    @classmethod
    def stat_array_dataset(cls, dataset):
        """Statistic function value per bin given the current model."""
        model = dataset.flux_pred()
        data = dataset.data.dnde.quantity
        try:
            sigma = dataset.data.dnde_err.quantity
        except AttributeError:
            sigma = (dataset.data.dnde_errn + dataset.data.dnde_errp).quantity / 2
        return ((data - model) / sigma).to_value("") ** 2


class Chi2AsymmetricErrorFitStatistic(FitStatistic):
    """Pseudo-Chi2 fit statistic class for measurements with gaussian asymmetric errors with upper limits.

    Assumes that regular data follow asymmetric normal pdf and upper limits follow complementary error functions
    """

    @classmethod
    def stat_array_dataset(cls, dataset):
        """Estimate statistic from probability distributions,
        assumes that flux points correspond to asymmetric gaussians
        and upper limits complementary error functions.
        """
        model = np.zeros(dataset.data.dnde.data.shape) + dataset.flux_pred().to_value(
            dataset.data.dnde.unit
        )

        stat = np.zeros(model.shape)

        mask_valid = ~np.isnan(dataset.data.dnde.data)
        loc = dataset.data.dnde.data[mask_valid]
        value = model[mask_valid]
        try:
            mask_p = (model >= dataset.data.dnde.data)[mask_valid]
            scale = np.zeros(mask_p.shape)
            scale[mask_p] = dataset.data.dnde_errp.data[mask_valid][mask_p]
            scale[~mask_p] = dataset.data.dnde_errn.data[mask_valid][~mask_p]

            mask_invalid = np.isnan(scale)
            scale[mask_invalid] = dataset.data.dnde_err.data[mask_valid][mask_invalid]
        except AttributeError:
            scale = dataset.data.dnde_err.data[mask_valid]

        stat[mask_valid] = ((value - loc) / scale) ** 2

        mask_ul = dataset.data.is_ul.data
        value = model[mask_ul]
        loc_ul = dataset.data.dnde_ul.data[mask_ul]
        scale_ul = dataset.data.dnde_ul.data[mask_ul]
        stat[mask_ul] = 2 * np.log(
            (erfc((loc_ul - value) / scale_ul) / 2)
            / (erfc((loc_ul - 0) / scale_ul) / 2)
        )

        stat[np.isnan(stat.data)] = 0
        return stat


class ProfileFitStatistic(FitStatistic):
    """Pseudo-Chi2 fit statistic class for measurements with gaussian asymmetric errors with upper limits.

    Assumes that regular data follow asymmetric normal pdf and upper limits follow complementary error functions
    """

    @classmethod
    def stat_array_dataset(cls, dataset):
        """Estimate statitistic from interpolation of the likelihood profile."""
        model = np.zeros(dataset.data.dnde.data.shape) + (
            dataset.flux_pred() / dataset.data.dnde_ref
        ).to_value("")
        stat = np.zeros(model.shape)
        for idx in np.ndindex(dataset._profile_interpolators.shape):
            stat[idx] = dataset._profile_interpolators[idx](model[idx])
        return stat


class FitStatisticPenalty:
    """Base class for fit statistic penalties.

    Parameters
    ----------
    parameters : list of `~gammapy.modeling.Parameter`
        List of parameters to apply the penalty to.
    lambda_ : float
        Regularization strength (Lagrange multiplier).
    """

    def __init__(self, parameters, lambda_=1.0):
        self.parameters = parameters  # can we keep it here? Is it safe?
        self.lambda_ = lambda_

    def stat_sum(self):
        """Compute the penalty term."""
        raise NotImplementedError


class GaussianPriorPenalty(FitStatisticPenalty):
    """Penalty based on a multivariate Gaussian prior.

    This implements a quadratic penalty of the form:

        lambda * (x - mean)^T C**-1 (x - mean)

    where x are the parameter values, μ is the prior mean vector,
    and C is the prior covariance matrix.

    If C is the identity matrix (default), this is equivalent to the L2 (ridge) regression.

    Parameters
    ----------
    parameters : list of `~gammapy.modeling.Parameter`
        Parameters to which the penalty is applied.
    mean : list of float, optional
        Prior mean values for each parameter. If not provided, defaults to zeros.
    covariance : array-like, optional
        Prior covariance matrix. If not provided, an identity matrix is used.
    lambda_ : float
        Regularization strength (Lagrange multiplier).
    """

    def __init__(self, parameters, mean=None, covariance=None, lambda_=1.0):
        super().__init__(parameters, lambda_)
        self.mean = np.array(mean) if mean is not None else np.zeros(len(parameters))
        self.covariance = (
            covariance if covariance is not None else np.eye(len(parameters))
        )

    @property
    def covariance(self):
        """Return covariance matrix of the multivariate gaussian."""
        return self._covariance

    @covariance.setter
    def covariance(self, matrix):
        """Set covariance matrix."""
        from gammapy.modeling import Covariance

        self._covariance = Covariance(self.parameters, matrix)
        self._inverse_covariance = np.linalg.inv(self._covariance.data)

    def stat_sum(self):
        """Compute the Gaussian prior penalty."""
        x = np.array([p.value for p in self.parameters])
        delta = x - self.mean
        penalty = np.dot(delta, self._inverse_covariance @ delta)

        return self.lambda_ * penalty

    @classmethod
    def from_precision(cls, parameters, precision, mean=None, lambda_=1.0):
        """Create a GaussianPriorPenalty from a precision matrix (inverse covariance).

        Internally uses `~scipy.stats.Covariance.from_precision`

        Parameters
        ----------
        parameters : list of `~gammapy.modeling.Parameter` or `~gammapy.modeling.Parameters`
            Parameters to which the penalty is applied.
        precision : `~numpy.ndarray`
            Inverse covariance matrix (precision matrix).
        mean : `~numpy.ndarray` or None
            Mean values. If None, uses zero array. Default is None.
        lambda_ : float
            Penalty strength (Lagrange multiplier).
        """
        from scipy.stats import Covariance

        cov = Covariance.from_precision(precision)
        mean = mean if mean is not None else np.zeros(len(parameters))
        return cls(
            parameters=parameters, mean=mean, covariance=cov.covariance, lambda_=lambda_
        )

    @classmethod
    def from_diagonal(cls, parameters, sigma, mean=None, lambda_=1.0):
        """Create a GaussianPriorPenalty with a diagonal covariance matrix.

        Parameters
        ----------
        parameters : list of Parameter
            Parameters to apply the prior to.
        sigma : ~numpy.ndarray` or float
            Standard deviation(s) for each parameter.
        mean : `~numpy.ndarray` or float, optional
            Mean values. If None, uses zero array. Default is None.
        lambda_ : float
            Penalty scaling factor.
        """
        from scipy.stats import Covariance

        sigma = np.broadcast_to(sigma, len(parameters))

        mean = mean if mean is not None else 0
        mean = np.broadcast_to(mean, len(parameters))

        covariance = Covariance.from_diagonal(sigma**2)
        return cls(
            parameters=parameters,
            mean=mean,
            covariance=covariance.covariance,
            lambda_=lambda_,
        )

    @classmethod
    def L2_penalty(cls, parameters, mean=None, lambda_=1.0):
        """Standard ridge (L2) regularization penalty.

        Computes a penalty of the form:  lambda * sum ||x_i - mu_i||²
        which corresponds to applying an identity covariance gaussian prior.

        Parameters
        ----------
        parameters : list of `~gammapy.modeling.Parameter`
            Parameters to which the penalty is applied.
        mean : list of float, optional
            Prior mean values for each parameter. If not provided, defaults to zeros.
        lambda_ : float
            Regularization strength (Lagrange multiplier).
        """
        return cls.from_diagonal(parameters, sigma=1, mean=mean, lambda_=lambda_)

    @classmethod
    def SmoothnessPenalty(cls, parameters, lambda_=1.0):
        """Create a smoothness penalty using finite differences.

        Parameters
        ----------
        parameters: list of `~gammapy.modeling.Parameter`
            Parameters to which the penalty is applied.
        lambda_: float, optional
            Penalty strength. Default is 1.
        """
        from scipy.sparse import diags

        n_params = len(parameters)
        diagonals = [
            2 * np.ones(n_params),  # main diagonal
            -1 * np.ones(n_params - 1),  # upper diagonal
            -1 * np.ones(n_params - 1),  # lower diagonal
        ]
        Q = diags(diagonals, [0, -1, 1]).toarray()
        return cls.from_precision(parameters, Q, mean=0, lambda_=lambda_)

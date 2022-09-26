# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
from scipy.stats import chi2
from gammapy.utils.roots import find_roots
from .fit_statistics import cash, wstat

__all__ = ["WStatCountsStatistic", "CashCountsStatistic"]


class CountsStatistic(abc.ABC):
    """Counts statistics base class"""

    @property
    @abc.abstractmethod
    def stat_null(self):
        pass

    @property
    @abc.abstractmethod
    def stat_max(self):
        pass

    @property
    @abc.abstractmethod
    def n_sig(self):
        pass

    @property
    @abc.abstractmethod
    def n_bkg(self):
        pass

    @property
    @abc.abstractmethod
    def error(self):
        pass

    @abc.abstractmethod
    def _stat_fcn(self):
        pass

    @property
    def ts(self):
        """Return stat difference (TS) of measured excess versus no excess."""
        # Remove (small) negative TS due to error in root finding
        ts = np.clip(self.stat_null - self.stat_max, 0, None)
        return ts

    @property
    def sqrt_ts(self):
        """Return statistical significance of measured excess.
        The sign of the excess is applied to distinguish positive and negative fluctuations.
        """
        return np.sign(self.n_sig) * np.sqrt(self.ts)

    @property
    def p_value(self):
        """Return p_value of measured excess.
        Here the value accounts only for the positive excess significance (i.e. one-sided).
        """
        return 0.5 * chi2.sf(self.ts, 1)

    def compute_errn(self, n_sigma=1.0):
        """Compute downward excess uncertainties.

        Searches the signal value for which the test statistics is n_sigma**2 away from the maximum.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the uncertainty expressed in number of sigma. Default is 1.
        """
        errn = np.zeros_like(self.n_sig, dtype="float")
        min_range = self.n_sig - 2 * n_sigma * (self.error + 1)

        it = np.nditer(errn, flags=["multi_index"])
        while not it.finished:
            roots, res = find_roots(
                self._stat_fcn,
                min_range[it.multi_index],
                self.n_sig[it.multi_index],
                nbin=1,
                args=(self.stat_max[it.multi_index] + n_sigma**2, it.multi_index),
            )
            if np.isnan(roots[0]):
                errn[it.multi_index] = self.n_on[it.multi_index]
            else:
                errn[it.multi_index] = self.n_sig[it.multi_index] - roots[0]
            it.iternext()

        return errn

    def compute_errp(self, n_sigma=1):
        """Compute upward excess uncertainties.

        Searches the signal value for which the test statistics is n_sigma**2 away from the maximum.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the uncertainty expressed in number of sigma. Default is 1.
        """

        errp = np.zeros_like(self.n_on, dtype="float")
        max_range = self.n_sig + 2 * n_sigma * (self.error + 1)

        it = np.nditer(errp, flags=["multi_index"])
        while not it.finished:
            roots, res = find_roots(
                self._stat_fcn,
                self.n_sig[it.multi_index],
                max_range[it.multi_index],
                nbin=1,
                args=(self.stat_max[it.multi_index] + n_sigma**2, it.multi_index),
            )
            errp[it.multi_index] = roots[0]
            it.iternext()

        return errp - self.n_sig

    def compute_upper_limit(self, n_sigma=3):
        """Compute upper limit on the signal.

        Searches the signal value for which the test statistics is n_sigma**2 away from the maximum
        or from 0 if the measured excess is negative.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the upper limit expressed in number of sigma. Default is 3.
        """

        ul = np.zeros_like(self.n_on, dtype="float")

        min_range = self.n_sig
        max_range = self.n_sig + 2 * n_sigma * (self.error + 1)
        it = np.nditer(ul, flags=["multi_index"])

        while not it.finished:
            ts_ref = self._stat_fcn(min_range[it.multi_index], 0.0, it.multi_index)

            roots, res = find_roots(
                self._stat_fcn,
                min_range[it.multi_index],
                max_range[it.multi_index],
                nbin=1,
                args=(ts_ref + n_sigma**2, it.multi_index),
            )
            ul[it.multi_index] = roots[0]
            it.iternext()
        return ul

    @abc.abstractmethod
    def _n_sig_matching_significance_fcn(self):
        pass

    def n_sig_matching_significance(self, significance):
        """Compute excess matching a given significance.

        This function is the inverse of `significance`.

        Parameters
        ----------
        significance : float
            Significance

        Returns
        -------
        n_sig : `numpy.ndarray`
            Excess
        """

        n_sig = np.zeros_like(self.n_bkg, dtype="float")
        it = np.nditer(n_sig, flags=["multi_index"])

        while not it.finished:
            lower_bound = np.sqrt(self.n_bkg[it.multi_index]) * significance
            # find upper bounds for secant method as in scipy
            eps = 1e-4
            upper_bound = lower_bound * (1 + eps)
            upper_bound += eps if upper_bound >= 0 else -eps
            roots, res = find_roots(
                self._n_sig_matching_significance_fcn,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                args=(significance, it.multi_index),
                nbin=1,
                method="secant",
            )
            n_sig[it.multi_index] = roots[0]  # return NaN if fail
            it.iternext()
        return n_sig

    @abc.abstractmethod
    def sum(self, axis=None):
        """Return summed CountsStatistics.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
             Axis or axes on which to perform the summation.
             Default, axis=None, will perform the sum over the whole array.

        Returns
        -------
        stat : `~gammapy.stats.CountsStatistics`
             the return stat object
        """
        pass


class CashCountsStatistic(CountsStatistic):
    """Class to compute statistics for Poisson distributed variable with known background.

    Parameters
    ----------
    n_on : int
        Measured counts
    mu_bkg : float
        Known level of background
    """

    def __init__(self, n_on, mu_bkg):
        self.n_on = np.asanyarray(n_on)
        self.mu_bkg = np.asanyarray(mu_bkg)

    @property
    def n_bkg(self):
        """Expected background counts"""
        return self.mu_bkg

    @property
    def n_sig(self):
        """Excess"""
        return self.n_on - self.n_bkg

    @property
    def error(self):
        """Approximate error from the covariance matrix."""
        return np.sqrt(self.n_on)

    @property
    def stat_null(self):
        """Stat value for null hypothesis, i.e. 0 expected signal counts"""
        return cash(self.n_on, self.mu_bkg + 0)

    @property
    def stat_max(self):
        """Stat value for best fit hypothesis, i.e. expected signal mu = n_on - mu_bkg"""
        return cash(self.n_on, self.n_on)

    def _stat_fcn(self, mu, delta=0, index=None):
        return cash(self.n_on[index], self.mu_bkg[index] + mu) - delta

    def _n_sig_matching_significance_fcn(self, n_sig, significance, index):
        TS0 = cash(n_sig + self.mu_bkg[index], self.mu_bkg[index])
        TS1 = cash(n_sig + self.mu_bkg[index], self.mu_bkg[index] + n_sig)
        return np.sign(n_sig) * np.sqrt(np.clip(TS0 - TS1, 0, None)) - significance

    def sum(self, axis=None):
        n_on = self.n_on.sum(axis=axis)
        bkg = self.n_bkg.sum(axis=axis)
        return CashCountsStatistic(n_on=n_on, mu_bkg=bkg)

    def __getitem__(self, key):
        return CashCountsStatistic(n_on=self.n_on[key], mu_bkg=self.n_bkg[key])


class WStatCountsStatistic(CountsStatistic):
    """Class to compute statistics for Poisson distributed variable with unknown background.

    Parameters
    ----------
    n_on : int
        Measured counts in on region
    n_off : int
        Measured counts in off region
    alpha : float
        Acceptance ratio of on and off measurements
    mu_sig : float
        Expected signal counts in on region
    """

    def __init__(self, n_on, n_off, alpha, mu_sig=None):
        self.n_on = np.asanyarray(n_on)
        self.n_off = np.asanyarray(n_off)
        self.alpha = np.asanyarray(alpha)

        if mu_sig is None:
            self.mu_sig = np.zeros_like(self.n_on)
        else:
            self.mu_sig = np.asanyarray(mu_sig)

    @property
    def n_bkg(self):
        """Known background computed alpha * n_off"""
        return self.alpha * self.n_off

    @property
    def n_sig(self):
        """Excess"""
        return self.n_on - self.n_bkg - self.mu_sig

    @property
    def error(self):
        """Approximate error from the covariance matrix."""
        return np.sqrt(self.n_on + self.alpha**2 * self.n_off)

    @property
    def stat_null(self):
        """Stat value for null hypothesis, i.e. mu_sig expected signal counts"""
        return wstat(self.n_on, self.n_off, self.alpha, self.mu_sig)

    @property
    def stat_max(self):
        """Stat value for best fit hypothesis

        i.e. expected signal mu = n_on - alpha * n_off - mu_sig
        """
        return wstat(self.n_on, self.n_off, self.alpha, self.n_sig + self.mu_sig)

    def _stat_fcn(self, mu, delta=0, index=None):
        return (
            wstat(
                self.n_on[index],
                self.n_off[index],
                self.alpha[index],
                (mu + self.mu_sig[index]),
            )
            - delta
        )

    def _n_sig_matching_significance_fcn(self, n_sig, significance, index):
        stat0 = wstat(
            n_sig + self.n_bkg[index], self.n_off[index], self.alpha[index], 0
        )
        stat1 = wstat(
            n_sig + self.n_bkg[index],
            self.n_off[index],
            self.alpha[index],
            n_sig,
        )
        return np.sign(n_sig) * np.sqrt(np.clip(stat0 - stat1, 0, None)) - significance

    def sum(self, axis=None):
        n_on = self.n_on.sum(axis=axis)
        n_off = self.n_off.sum(axis=axis)
        alpha = self.n_bkg.sum(axis=axis) / n_off
        return WStatCountsStatistic(n_on=n_on, n_off=n_off, alpha=alpha)

    def __getitem__(self, key):
        return WStatCountsStatistic(
            n_on=self.n_on[key], n_off=self.n_off[key], alpha=self.alpha[key]
        )

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
from scipy.optimize import brentq, newton
from scipy.stats import chi2
from .fit_statistics import cash, get_wstat_mu_bkg, wstat

__all__ = ["WStatCountsStatistic", "CashCountsStatistic"]


class CountsStatistic(abc.ABC):
    @property
    def ts(self):
        """Return stat difference (TS) of measured excess versus no excess."""
        # Remove (small) negative TS due to error in root finding
        ts = np.clip(self.stat_null - self.stat_max, 0, None)
        return ts

    @property
    def sqrt_ts(self):
        """Return statistical significance of measured excess."""
        return np.sign(self.n_sig) * np.sqrt(self.ts)

    @property
    def p_value(self):
        """Return p_value of measured excess."""
        return chi2.sf(self.ts, 1)

    def compute_errn(self, n_sigma=1.0):
        """Compute downward excess uncertainties.

        Searches the signal value for which the test statistics is n_sigma**2 away from the maximum.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the uncertainty expressed in number of sigma. Default is 1.
        """
        errn = np.zeros_like(self.n_on, dtype="float")
        min_range = self.n_sig - 2 * n_sigma * (self.error + 1)

        it = np.nditer(errn, flags=["multi_index"])
        while not it.finished:
            try:
                res = brentq(
                    self._stat_fcn,
                    min_range[it.multi_index],
                    self.n_sig[it.multi_index],
                    args=(self.stat_max[it.multi_index] + n_sigma ** 2, it.multi_index),
                )
                errn[it.multi_index] = res - self.n_sig[it.multi_index]
            except ValueError:
                errn[it.multi_index] = -self.n_on[it.multi_index]
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
            errp[it.multi_index] = brentq(
                self._stat_fcn,
                self.n_sig[it.multi_index],
                max_range[it.multi_index],
                args=(self.stat_max[it.multi_index] + n_sigma ** 2, it.multi_index),
            )
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

        min_range = np.maximum(0, self.n_sig)
        max_range = min_range + 2 * n_sigma * (self.error + 1)
        it = np.nditer(ul, flags=["multi_index"])

        while not it.finished:
            TS_ref = self._stat_fcn(min_range[it.multi_index], 0.0, it.multi_index)

            ul[it.multi_index] = brentq(
                self._stat_fcn,
                min_range[it.multi_index],
                max_range[it.multi_index],
                args=(TS_ref + n_sigma ** 2, it.multi_index),
            )
            it.iternext()

        return ul

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
            try:
                n_sig[it.multi_index] = newton(
                    self._n_sig_matching_significance_fcn,
                    np.sqrt(self.n_bkg[it.multi_index]) * significance,
                    args=(significance, it.multi_index),
                )
            except:
                n_sig[it.multi_index] = np.nan

            it.iternext()
        return n_sig


class CashCountsStatistic(CountsStatistic):
    """Class to compute statistics (significance, asymmetric errors , ul) for Poisson distributed variable
    with known background.

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


class WStatCountsStatistic(CountsStatistic):
    """Class to compute statistics (significance, asymmetric errors , ul) for Poisson distributed variable
    with unknown background.

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
        """Excess """
        return self.n_on - self.n_bkg - self.mu_sig

    @property
    def error(self):
        """Approximate error from the covariance matrix."""
        return np.sqrt(self.n_on + self.alpha ** 2 * self.n_off)

    @property
    def stat_null(self):
        """Stat value for null hypothesis, i.e. mu_sig expected signal counts"""
        return wstat(self.n_on, self.n_off, self.alpha, self.mu_sig)

    @property
    def stat_max(self):
        """Stat value for best fit hypothesis, i.e. expected signal mu = n_on - alpha * n_off - mu_sig"""
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
            n_sig + self.n_bkg[index], self.n_off[index], self.alpha[index], n_sig,
        )
        return np.sign(n_sig) * np.sqrt(np.clip(stat0 - stat1, 0, None)) - significance

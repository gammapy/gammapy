# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
from scipy.optimize import brentq
from gammapy.stats import wstat, cash

__all__ = ["WStatEstimator", "CashEstimator"]

class FitStatisticEstimator(abc.ABC):
    @property
    def significance(self):
        """Return statistical significance of measured excess."""
        return np.sign(self.excess)*np.sqrt(self.TS_null-self.TS_max)

    def compute_errn(self, n_sigma=1):
        """Compute downward excess uncertainties.

        Searches the signal value for which the test statistics is n_sigma away from the maximum.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the uncertainty expressed in number of sigma. Default is 1.
        """
        min_range = self.excess - 2 * n_sigma * self.std

        try:
            errn = brentq(
                self._stat_fcn,
                min_range,
                self.excess,
                args=(self.TS_max + n_sigma))
        except ValueError:
            return -self.n_on

        return errn - self.excess

    def compute_errp(self, n_sigma=1):
        """Compute upward excess uncertainties.

        Searches the signal value for which the test statistics is n_sigma away from the maximum.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the uncertainty expressed in number of sigma. Default is 1.
        """
        max_range = self.excess + 2 * n_sigma * self.std

        errp = brentq(
            self._stat_fcn,
            self.excess,
            max_range,
            args=(self.TS_max + n_sigma))

        return errp - self.excess

    def compute_upper_limit(self, n_sigma=3):
        """Compute upper limit on the signal.

        Searches the signal value for which the test statistics is n_sigma away from the maximum
        or from 0 if the measured excess is negative.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the upper limit expressed in number of sigma. Default is 3.
        """
        min_range = np.maximum(0, self.excess)
        max_range = min_range + 2 * n_sigma * self.std
        TS_ref = self._stat_fcn(min_range,0.)
        return brentq(
            self._stat_fcn,
            min_range,
            max_range,
            args=(TS_ref + n_sigma))


class CashEstimator(FitStatisticEstimator):
    """Class to compute statistics (significance, asymmetric errors , ul) for Poisson distributed variable
    with known background.

    Parameters
    ----------
    n_on : int
        Measured counts
    mu_bkg : float
        Expected level of background
    """
    def __init__(self, n_on, mu_bkg):
        self.n_on = n_on
        self.mu_bkg= mu_bkg

    @property
    def excess(self):
        return self.n_on - self.mu_bkg

    @property
    def std(self):
        """Approximate error."""
        return np.sqrt(self.n_on+1)

    @property
    def TS_null(self):
        """Stat value for null hypothesis, i.e. 0 expected signal counts"""
        return cash(self.n_on, self.mu_bkg + 0)

    @property
    def TS_max(self):
        """Stat value for best fit hypothesis, i.e. expected signal mu = n_on - mu_bkg"""
        return cash(self.n_on, self.n_on)

    def _stat_fcn(self, mu, delta):
        return cash(self.n_on, self.mu_bkg + mu) - delta


class WStatEstimator(FitStatisticEstimator):
    """Class to compute statistics (significance, asymmetric errors , ul) for Poisson distributed variable
    with unknown background.

    Parameters
    ----------
    n_on : int
        Measured counts in signal (ON) region
    n_off : int
        Measured counts in background only (OFF) region
    alpha : float
        Acceptance ratio of ON and OFF measurements
    """

    def __init__(self, n_on, n_off, alpha):
        self.n_on = n_on
        self.n_off = n_off
        self.alpha = alpha

    @property
    def excess(self):
        return self.n_on - self.alpha * self.n_off

    @property
    def std(self):
        return np.sqrt(self.n_on + self.alpha ** 2 * self.n_off)

    @property
    def TS_null(self):
        """Stat value for null hypothesis, i.e. 0 expected signal counts"""
        return wstat(self.n_on, self.n_off, self.alpha, 0)

    @property
    def TS_max(self):
        """Stat value for best fit hypothesis, i.e. expected signal mu = n_on - alpha * n_off"""
        return wstat(self.n_on, self.n_off, self.alpha, self.excess)

    def _stat_fcn(self, mu, delta):
        return wstat(self.n_on, self.n_off, self.alpha, mu) - delta


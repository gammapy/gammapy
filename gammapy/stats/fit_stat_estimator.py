# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
import scipy
from scipy.optimize import brentq
from gammapy.stats import wstat, cash

__all__ = ["WStatEvaluator", "CashEvaluator"]


class FitStatisticEvaluator(abc.ABC):
    @property
    def significance(self):
        """Return statistical significance of measured excess."""
        return np.sign(self.excess) * np.sqrt(self.TS_null - self.TS_max)

    def compute_errn(self, n_sigma=1):
        """Compute downward excess uncertainties.

        Searches the signal value for which the test statistics is n_sigma away from the maximum.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the uncertainty expressed in number of sigma. Default is 1.
        """
        errn = np.zeros_like(self.n_on, dtype="float")
        min_range = self.excess - 2 * n_sigma * self.std

        it = np.nditer(errn, flags=["multi_index"])
        while not it.finished:
            try:
                res = brentq(
                    self._stat_fcn,
                    min_range[it.multi_index],
                    self.excess[it.multi_index],
                    args=(self.TS_max[it.multi_index] + n_sigma, it.multi_index),
                )
                errn[it.multi_index] = res - self.excess[it.multi_index]
            except ValueError:
                errn[it.multi_index] = -self.n_on[it.multi_index]
            it.iternext()

        return errn

    def compute_errp(self, n_sigma=1):
        """Compute upward excess uncertainties.

        Searches the signal value for which the test statistics is n_sigma away from the maximum.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the uncertainty expressed in number of sigma. Default is 1.
        """
        errp = np.zeros_like(self.n_on, dtype="float")
        max_range = self.excess + 2 * n_sigma * self.std

        it = np.nditer(errp, flags=["multi_index"])
        while not it.finished:
            errp[it.multi_index] = brentq(
                self._stat_fcn,
                self.excess[it.multi_index],
                max_range[it.multi_index],
                args=(self.TS_max[it.multi_index] + n_sigma, it.multi_index),
            )
            it.iternext()

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
        ul = np.zeros_like(self.n_on, dtype="float")

        min_range = np.maximum(0, self.excess)
        max_range = min_range + 2 * n_sigma * self.std
        it = np.nditer(ul, flags=["multi_index"])

        while not it.finished:
            TS_ref = self._stat_fcn(min_range[it.multi_index], 0.0, it.multi_index)

            ul[it.multi_index] = brentq(
                self._stat_fcn,
                min_range[it.multi_index],
                max_range[it.multi_index],
                args=(TS_ref + n_sigma, it.multi_index),
            )
            it.iternext()

        return ul


class CashEvaluator(FitStatisticEvaluator):
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
        self.n_on = np.asanyarray(n_on)
        self.mu_bkg = np.asanyarray(mu_bkg)

    @property
    def excess(self):
        return self.n_on - self.mu_bkg

    @property
    def std(self):
        """Approximate error."""
        return np.sqrt(self.n_on + 1)

    @property
    def TS_null(self):
        """Stat value for null hypothesis, i.e. 0 expected signal counts"""
        return cash(self.n_on, self.mu_bkg + 0)

    @property
    def TS_max(self):
        """Stat value for best fit hypothesis, i.e. expected signal mu = n_on - mu_bkg"""
        return cash(self.n_on, self.n_on)

    def _stat_fcn(self, mu, delta=0, index=None):
        return cash(self.n_on[index], self.mu_bkg[index] + mu) - delta

    @property
    def _significance_direct(self):
        """Compute significance directly via Poisson probability.

        Reference: TODO (is this ever used?)
        """
        # Compute tail probability to see n_on or more counts
        # Note that we're using ``k = n_on - 1`` to get the probability
        # for n_on included or more, because `poisson.sf(k)` returns the
        # probability for more than k, with k excluded
        # For `n_on = 0` this returns `
        probability = scipy.stats.poisson.sf(self.n_on - 1, self.mu_bkg)

        # Convert probability to a significance
        return scipy.stats.norm.isf(probability)

class WStatEvaluator(FitStatisticEvaluator):
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
        self.n_on = np.asanyarray(n_on)
        self.n_off = np.asanyarray(n_off)
        self.alpha = np.asanyarray(alpha)

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

    def _stat_fcn(self, mu, delta=0, index=None):
        return wstat(self.n_on[index], self.n_off[index], self.alpha[index], mu) - delta

    @property
    def _significance_direct(self):
        """Compute significance directly via Poisson probability.

        Reference: https://ui.adsabs.harvard.edu/abs/1993NIMPA.328..570A

        You can use this method for small n_on < 10.
        In this case the Li & Ma formula isn't correct any more.
        """
        f = np.math.factorial
        probability = np.ones_like(self.n_on, dtype='float64')

        it = np.nditer(probability, flags=["multi_index"])
        while not it.finished:
            n_on = int(self.n_on[it.multi_index])
            n_off = int(self.n_off[it.multi_index])
            alpha = self.alpha[it.multi_index]
            # Compute tail probability to see n_on or more counts
            for n in range(0, n_on):
                term_1 = alpha ** n / (1 + alpha) ** (n_off + n + 1)
                term_2 = f(n_off + n) / (f(n) * f(n_off))
                probability[it.multi_index] -= term_1 * term_2
            it.iternext()

        # Convert probability to a significance
        return scipy.stats.norm.isf(probability)

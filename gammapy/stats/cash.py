# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy.optimize import brentq
from gammapy.stats.fit_statistics import cash

__all__ = ["CashEstimator"]

class CashEstimator:
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
        """Stat value for null hypothesis, i.e. 0 counts"""
        return cash(self.n_on, self.mu_bkg + 0)

    @property
    def TS_max(self):
        """Stat value for best fit hypothesis, i.e. mu = n_on - mu_bkg"""
        return cash(self.n_on, self.n_on)

    @property
    def significance(self):
        return np.sign(self.excess)*np.sqrt(self.TS_null-self.TS_max)

    def _cash_fcn(self, mu, delta):
        return cash(self.n_on, self.mu_bkg + mu) - delta

    def compute_errn(self, n_sigma=1):
        min_range = np.maximum(-self.mu_bkg, self.excess - 2 * n_sigma * self.std)
        try:
            errn = brentq(
                self._cash_fcn,
                min_range,
                self.excess,
                args=(self.TS_max + n_sigma)
            )
        except ValueError:
            errn = -self.mu_bkg

        return errn - self.excess

    def compute_errp(self, n_sigma=1):
        max_range = self.excess + 2 * n_sigma * self.std

        errp = brentq(
            self._cash_fcn,
            self.excess,
            max_range,
            args=(self.TS_max + n_sigma)
        )

        return errp - self.excess

    def compute_upper_limit(self, n_sigma=3):
        min_range = np.maximum(0, self.excess)
        max_range = min_range + 2 * n_sigma * self.std
        ref = cash(self.n_on, self.mu_bkg + min_range)
        return brentq(
            self._cash_fcn,
            min_range,
            max_range,
            args=(ref + n_sigma))

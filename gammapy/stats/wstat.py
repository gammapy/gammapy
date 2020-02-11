# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy.optimize import brentq
from gammapy.stats import wstat

__all__ = ["WStatEstimator"]

class WStatEstimator:
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
        return wstat(self.n_on, self.n_off, self.alpha, 0)

    @property
    def TS_max(self):
        return wstat(self.n_on, self.n_off, self.alpha, self.excess)

    @property
    def significance(self):
        return np.sign(self.excess)*np.sqrt(self.TS_null-self.TS_max)

    @staticmethod
    def _wstat_fcn(self, mu, delta):
        return wstat(self.n_on, self.n_off, self.alpha, mu) - delta

    def compute_errn(self, n_sigma=1):
        min_range = self.excess - 2 * n_sigma * self.std

        errn = brentq(
            self._wstat_fcn,
            min_range,
            self.excess,
            args=(self.TS_max + n_sigma))

        return errn - self.excess

    def compute_errp(self, n_sigma=1):
        max_range = self.excess + 2 * n_sigma * self.std

        errp = brentq(
            self._wstat_fcn,
            self.excess,
            max_range,
            args=(self.TS_max + n_sigma))

        return errp - self.excess

    def compute_upper_limit(self, n_sigma=3):
        min_range = np.maximum(0, self.excess)
        max_range = min_range + 2 * n_sigma * self.std
        return brentq(
            self._wstat_fcn,
            min_range,
            max_range,
            args=(self.TS_max + n_sigma))




# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np

__all__ = ['EnergyAxis', 'log_mean_energy']

class EnergyAxis(object):
    """Log(E) axis"""
    def __init__(self, e):
        self.e = e
        self.log_e = np.log10(e)

    def __call__(self, e):
        try:
            z1 = np.where(e >= self.e)[0][-1]
        except ValueError:
            # Loop over es by hand
            z1 = np.empty_like(e, dtype=int)
            for ii in range(e.size):
                # print ii, e[ii], np.where(e[ii] >= self.e)
                z1[ii] = np.where(e[ii] >= self.e)[0][-1]
        z2 = z1 + 1
        e1 = self.e[z1]
        e2 = self.e[z2]
        # print e1, '<=', e, '<', e2
        return z1, z2, e1, e2


def log_mean_energy(e1, e2):
    """Compute log arithmetic mean energy"""
    log_e1, log_e2 = np.log(e1), np.log(e2)
    log_e = 0.5 * (log_e1 + log_e2)
    e = np.exp(log_e)
    return e

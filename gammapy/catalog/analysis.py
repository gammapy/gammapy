# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog analysis methods."""
from __future__ import print_function, division
import numpy as np

__all__ = ['FluxDistribution']


class FluxDistribution(object):
    """Catalog flux distribution analysis and plotting.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Input catalog. Must contain the following columns:

        * Source flux column called 'S'
        * Source extension column called 'r'
    """

    def __init__(self, table, label):
        self.table = table.copy()
        self.table.sort('S')
        self.label = label

    def print_info(self):
        S = self.table['S']
        print('Flux min: {0}'.format(S.min()))
        print('Flux max: {0}'.format(S.max()))

    def plot_integral_count(self):
        import matplotlib.pyplot as plt
        N_min = 0.8

        # Make lists of flux, integral count and
        # add one point at the right edge to make
        # the plot look like a histogram there.
        S = list(self.table['S'])
        N = range(len(S), 0, -1)

        S.append(S[-1])
        N.append(N_min)

        plt.step(x=S, y=N, where='pre', label=self.label, lw=2, alpha=0.8)
        plt.xlabel('Flux (%Crab)')
        plt.ylabel('Integral Count (>Flux)')
        plt.loglog()
        #plt.ylim(N_min, 1.2 * len(S))
        #plt.legend()

    def plot_differential_count(self):
        raise NotImplementedError

    def plot_integral_flux(self):
        import matplotlib.pyplot as plt
        S = self.table['S']
        y = 0.01 * np.cumsum(S[::-1])[::-1]

        plt.step(x=S, y=y, where='pre', label=self.label, lw=2, alpha=0.8)
        plt.xlabel('Flux (%Crab)')
        plt.ylabel('Integral Flux (>Flux) in Crab')
        plt.semilogx()

    def plot_differential_flux(self):
        raise NotImplementedError

    def fit_power_law(self, S_min, S_max):
        raise NotImplementedError

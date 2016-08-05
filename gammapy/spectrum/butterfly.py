# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import QTable

__all__ = [
    'SpectrumButterfly',
]


class SpectrumButterfly(QTable):
    """Spectral model butterfly class.

    Columns:

    - ``energy``
    - ``flux_lo``
    - ``flux``
    - ``flux_hi``
    """

    def plot(self, energy_range=None, ax=None, energy_power=0, **kwargs):
        """Plot.

        ``kwargs`` are passed to ``matplotlib.pyplot.errorbar``.
        """
        if energy_range is None:
            energy_range = np.min(self['energy']), np.max(self['energy'])

        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        kwargs.setdefault('color', 'black')
        kwargs.setdefault('alpha', 0.2)
        kwargs.setdefault('linewidth', 0)

        x = self['energy']
        y_lo = self['flux_lo'] * np.power(x, energy_power)
        y_hi = self['flux_hi'] * np.power(x, energy_power)

        where = (y_hi > 0) & (x >= energy_range[0]) & (x <= energy_range[1])
        ax.fill_between(x.value, y_lo.value, y_hi.value, where=where, **kwargs)

        ax.set_xlabel('Energy [{}]'.format(self['energy'].unit))
        ax.set_ylabel('Flux [{}]'.format(y_lo.unit))
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        return ax

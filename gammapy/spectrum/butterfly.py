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

    def plot(self, ax=None, energy_power=2, **kwargs):
        """Plot.

        ``kwargs`` are passed to ``matplotlib.pyplot.errorbar``.
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        x = self['energy'].value
        y = self['flux'].value
        yerr = self['flux_lo'].value + self['flux_hi'].value

        y = y * np.power(x, energy_power)
        yerr = yerr * np.power(x, energy_power)

        ax.errorbar(x=x, y=y, yerr=yerr, **kwargs)

        kwargs.setdefault('capsize', 0)
        ax.errorbar(x, y, yerr=yerr, **kwargs)
        ax.set_xlabel('Energy [{}]'.format(self['energy'].unit))
        ax.set_ylabel('Flux [{}]'.format(self['flux'].unit))
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')

        return ax

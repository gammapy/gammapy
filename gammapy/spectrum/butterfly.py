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

    def plot(self, energy_range=None, ax=None, energy_power=0,
             energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1', **kwargs):
        """Plot.

        ``kwargs`` are passed to ``matplotlib.pyplot.fill_between``.
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        kwargs.setdefault('facecolor', 'black')
        kwargs.setdefault('alpha', 0.2)
        kwargs.setdefault('linewidth', 0)

        energy = self['energy'].to(energy_unit)
        flux_lo = self['flux_lo'].to(flux_unit)
        flux_hi = self['flux_hi'].to(flux_unit)
        y_lo = flux_lo * np.power(energy, energy_power)
        y_hi = flux_hi * np.power(energy, energy_power)

        eunit = [_ for _ in flux_lo.unit.bases if _.physical_type == 'energy'][0]
        y_lo = y_lo.to(eunit ** energy_power * flux_lo.unit)
        y_hi = y_hi.to(eunit ** energy_power * flux_hi.unit)

        if energy_range is None:
            energy_range = np.min(energy), np.max(energy)

        where = (y_hi > 0) & (energy >= energy_range[0]) & (energy <= energy_range[1])
        ax.fill_between(energy.value, y_lo.value, y_hi.value, where=where, **kwargs)

        ax.set_xlabel('Energy [{}]'.format(self['energy'].unit))
        if energy_power > 0:
            ax.set_ylabel('E{0} * Flux [{1}]'.format(energy_power, y_lo.unit))
        else:
            ax.set_ylabel('Flux [{}]'.format(y_lo.unit))
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        return ax

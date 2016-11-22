# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Lightcurve and elementary temporal functions
"""
from astropy.table import QTable
from astropy.units import Quantity
import astropy.units as u

__all__ = [
    'LightCurve',
]


class LightCurve(QTable):
    """LightCurve class.

    Contains all data in the tabular form with columns
    tstart, tstop, flux, flux error, time bin (opt).
    Possesses functions allowing plotting data, saving as txt
    and elementary stats like mean & std dev.

    TODO: specification of format is work in progress
    See https://github.com/open-gamma-ray-astro/gamma-astro-data-formats/pull/61
    """

    def plot(self, ax=None):
        """Plot flux versus time.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.
            If None, uses the current `~matplotlib.axes.Axes`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.
            If None, uses the current `~matplotlib.axes.Axes`.
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        tstart = self['TIME_MIN'].to('s')
        tstop = self['TIME_MAX'].to('s')
        time = (tstart + tstop) / 2.0
        flux = self['FLUX'].to('cm-2 s-1')
        errors = self['FLUX_ERR'].to('cm-2 s-1')

        ax.errorbar(time.value, flux.value, yerr=errors.value, linestyle="None")
        ax.scatter(time, flux)
        ax.set_xlabel("Time (secs)")
        ax.set_ylabel("Flux ($cm^{-2} sec^{-1}$)")

        return ax

    @classmethod
    def simulate_example(cls):
        """Simulate an example `LightCurve`.

        TODO: add options to simulate some more interesting lightcurves.
        """
        lc = cls()

        lc['TIME_MIN'] = [1, 4, 7, 9] * u.s
        lc['TIME_MAX'] = [1, 4, 7, 9] * u.s
        lc['FLUX'] = Quantity([1, 4, 7, 9], 'cm^-2 s^-1')
        lc['FLUX_ERR'] = Quantity([0.1, 0.4, 0.7, 0.9], 'cm^-2 s^-1')

        return lc

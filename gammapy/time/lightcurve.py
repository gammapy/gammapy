# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Lightcurve and elementary temporal functions
"""
from astropy.table import QTable
from astropy.units import Quantity
import astropy.units as u
import numpy as np

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

        ax.errorbar(time.value, flux.value,
                    yerr=errors.value, linestyle="None")
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

    def compute_fvar(self):
        """Calculate the fractional excess variance (Fvar) of `LightCurve`.
       
        The fractional excess variance :`Fvar` is given by

        .. math :: 
	    Fvar = \sqrt{\frac{S^{2} - \bar{\sigma^{2}}}{\bar{x}^{2}}}
	
	Fvar is an intrinsic variability estimator. It is excess variance 
	after accounting for the measurement errors on the light curve 
	:math: `\sigma`. S is the variance. 

	Parameters
	----------
        time_min : array_like
	    Start time for bin
        time_max : array_like
	    Stop time for bin
        flux : array_like
	    Flux values with units explicitly specified
        fluxerr : array_like
	    Flux error values with units explicitly specified
	
	Returns
	-------
	Fvar : array_like

        Reference 
	---------
	
	.. [1] On characterizing the variability properties of X-ray light curves from active galaxiesVaughan et al.,2003 
        http://adsabs.harvard.edu/abs/2003MNRAS.345.1271V
        """
        time_min = self['TIME_MIN']
        time_max = self['TIME_MAX']
        time = (time_min + time_max) / 2.0
        flux = self['FLUX'].value
        fluxerr = self['FLUX_ERR'].value
        nptsperbin = len(flux)
        phisum = np.sum(flux)
        phimean = phisum / nptsperbin
        Ssq = (np.sum((flux - phimean )**2)
               ) / (nptsperbin - 1.0)
        S = np.sqrt(Ssq)
        Sigsq = np.nansum(fluxerr**2) / nptsperbin
        fvar = np.sqrt(np.abs(Ssq - Sigsq)) / phimean
        fluxerrT = np.transpose(fluxerr)
        sigxserrA = np.sqrt(2.0 / nptsperbin) * (Sigsq / phimean)**2
        sigxserrB = np.sqrt(Sigsq / nptsperbin) * (2.0 * fvar / phimean)
        sigxserr = np.sqrt(sigxserrA**2 + sigxserrB**2)
        fvarerr = sigxserr / (2.0 * fvar)
        return fvar, fvarerr

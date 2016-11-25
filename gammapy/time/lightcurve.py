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

    def compute_fvar(self):
        """Calculate the fractional excess variance (Fvar) of `LightCurve`.

        """
        time_min = self['TIME_MIN']
        time_max = self['TIME_MAX']
	time = (time_min+time_max)/2.0
	flux = self['FLUX'].value
   	fluxerr = self['FLUX_ERR'].value
   	nptsperbin = len(flux)
   	phisum = np.sum(flux)
   	phimean = phisum / nptsperbin
  	#print nptsperbin,phisum,phimean     
   	unityrow = np.ones(nptsperbin)
   	Ssq = (np.sum( (flux[:]-phimean*unityrow[:])**2) ) / (nptsperbin-1.0)
   	S = np.sqrt(Ssq)
   	Sigsq = np.nansum(fluxerr[:]**2) / nptsperbin
   	#print fluxerr,np.nansum(fluxerr)#,nptsperbin        
   	fvar = np.sqrt(np.abs(Ssq-Sigsq)) / phimean
   	fluxerrT = np.transpose(fluxerr)
   	#ascii.write([fluxerrT],'simfluxerr.txt')#,np.sum(fluxerr),fluxerr#nptsperbin
   	sigxserrA = np.sqrt( 2.0/nptsperbin) * (Sigsq/phimean)**2
   	sigxserrB = np.sqrt(Sigsq/nptsperbin) * (2.0*fvar/phimean)
	#print sigxserrA,sigxserrB
	#import IPython; IPython.embed()
   	sigxserr = np.sqrt(sigxserrA**2 + sigxserrB**2)#*phimean
   	fvarerr = sigxserr / (2.0 * fvar)
   	#print sigxserrA,sigxserrB,sigxserr,fvar,fvarerr
	return fvar,fvarerr
  
     	

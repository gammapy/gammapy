# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Lightcurve and elementary temporal functions
"""
from astropy.table import Table
from astropy.table import QTable
from astropy.units import Quantity
import astropy.units as u


__all__ = [
    'LightCurve',
    'make_example_lightcurve',
]


class LightCurve(QTable):

    """ 
    LightCurve class  
            Contains all data in the tabular form with columns 
            tstart, tstop, flux, flux error, time bin (opt). 
            Possesses functions allowing plotting data, saving as txt 
            and elementary stats like mean & variance.
    units:
            time - secs
            flux - 1 / cm^2 sec
    """

#    def __init__(self, table):
#	super(LightCurve,self).__init__(table)    

    def lc_plot(self):
        """
        Plots the Lightcurve i.e. the flux as a function of time.
        Here, time for each bin is equal to the center of the bin, 
        i.e. the average of tstart and tstop	 
        """
        import matplotlib.pyplot as plt
        tstart = self['TIME_MIN'].to('s')
        tstop = self['TIME_MAX'].to('s')
        time = (tstart + tstop) / 2.0
        flux = self['FLUX'].to('cm-2 s-1')
        errors = self['FLUX_ERR'].to('cm-2 s-1')
        plt.errorbar(time.value, flux.value,
                     yerr=errors.value, linestyle="None")
        plt.scatter(time, flux)
        plt.xlabel("Time (secs)")
        plt.ylabel("Flux ($cm^{-2} sec^{-1}$)")
        plt.title("Lightcurve")

def make_example_lightcurve():
    """ Make an example lightcurve.
    """
    lc = LightCurve()
    lc['TIME_MIN'] = [1, 4, 7, 9] * u.s
    lc['TIME_MAX'] = [1, 4, 7, 9] * u.s
    lc['FLUX'] = Quantity([1, 4, 7, 9], 'cm^-2 s^-1')
    lc['FLUX_ERR'] = Quantity([0.1, 0.4, 0.7, 0.9], 'cm^-2 s^-1')
    return lc

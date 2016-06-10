# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Lightcurve and elementary temporal functions
"""
from astropy.table import Table

__all__ = ['LightCurve']


class LightCurve(Table):

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

    def __init__(self, table):
        self.table = table

    def flux_mean(self):
        """
        Computes the mean value of the flux in given time interval 
        """
        flux = self.table['FLUX']
        return flux.mean()

    def flux_std(self):
        """
        Computes the std deviation of the flux in given time interval 
        """
        flux = self.table['FLUX']
        return flux.std()

    def lc_plot(self):
        """
        Plots the Lightcurve i.e. the flux as a function of time.
        Here, time for each bin is equal to the center of the bin, 
        i.e. the average of tstart and tstop	 
        """
        import matplotlib.pyplot as plt
        tstart = self.table['TSTART'].to('s')
        tstop = self.table['TSTOP'].to('s')
        time = (tstart + tstop) / 2.0
        flux = self.table['FLUX'].to('cm-2 s-1')
        errors = self.table['ERRORS'].to('cm-2 s-1')
        plt.errorbar(time.value, flux.value,
                     yerr=errors.value, linestyle="None")
        plt.scatter(time, flux)
        plt.xlabel("Time (secs)")
        plt.ylabel("Flux ($cm^{-2} sec^{-1}$)")
        plt.title("Lightcurve")
        plt.show()

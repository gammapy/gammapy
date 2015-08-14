# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)

import numpy as np
from astropy import log
from ..spectrum.energy import *
import datetime
from astropy.io import fits

__all__ = ['CountsSpectrum']

class CountsSpectrum(object):
    """Counts spectrum dataset

    Parameters
    ----------

    counts: `~numpy.array`, list
        Counts
    energy: `~gammapy.spectrum.Energy`, `~gammapy.spectrum.EnergyBounds`.
        Energy axis

    Examples
    --------

    .. code-block:: python
    
        from gammapy.spectrum import *
        from gammapy.data import *
        ebounds = EnergyBounds.equal_log_spacing(1,10,10,'TeV') 
        counts = [6,3,8,4,9,5,9,5,5,1]
        spec = CountsSpectrum(counts, ebounds)
        hdu = spec.to_pha() 
    """

    def __init__(self, counts, energy, livetime=None):
        
        if np.asarray(counts).size != energy.nbins:
            raise ValueError("Dimension of {0} and {1} do not"
                             " match".format(counts, energy))
        
        if not isinstance(energy, Energy):
            raise ValueError("energy must be an Energy object")
        
        self.counts = np.asarray(counts)

        if isinstance (energy, EnergyBounds):
            self.energy = energy.log_centers
        else:
            self.energy = energy

        self._livetime = livetime

    @property
    def entries(self):
        """Total number of counts
        """
        return self.counts.sum()

    @property
    def livetime(self):
        """Live time of the dataset
        """
        return self._livetime

    @classmethod
    def from_fits(cls, hdu):
        """Read SPECTRUM fits extension from pha file (`CountsSpectrum`).

        Parameters
        ----------
        hdu: `~astropy.io.fits.BinTableHDU`
            ``SPECTRUM`` extensions.
        """
        pass

    @classmethod
    def from_eventlist(cls, event_list, bins):
        """Create CountsSpectrum from fits 'EVENTS' extension (`CountsSpectrum`).

        Subsets of the event list should be choosen via the approriate methods
        in `~gammapy.data.EventList`.

        Parameters
        ----------
        event_list: 
            `~astropy.io.fits.BinTableHDU, `gammapy.data.EventListDataSet`,
            `gammapy.data.EventList`, str (filename)
        bins: `gammapy.spectrum.energy.EnergyBounds`
            Energy bin edges
        """

        if isinstance (event_list,fit.BinTableHDU):
            event_list = EventList.read(event_list)
        elif isinstance(event_list, gammapy.data.EventListDataSet):
            event_list = event_list.event_list
        elif isinstance(event_list, str):
            event_list = EventList.read(event_list, hdu='EVENTS') 

        val = np.histogram(event_list.energy,bins)
        livetime = event_list.observation_live_time_duration

        return cls(val, energy, livetime)


    def to_pha(self, **kwargs):
        """Output OGIP pha file
        
        Returns
        -------
        pha : `~astropy.io.fits.BinTableHDU`
            PHA FITS HDU

        Notes
        -----
        For more info on the PHA FITS file format see:
        http://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/summary/
        ogip_92_007_summary.html
        """
        col1 = fits.Column(name="CHANNEL", array=self.energy.value,
                          format = 'E', unit = '{0}'.format(self.energy.unit))
        col2 = fits.Column(name="COUNTS", array=self.counts, format='E')
        
        cols = fits.ColDefs([col1, col2])

        hdu = fits.BinTableHDU.from_columns(cols)
        header = hdu.header

        # copied from np_to_pha
        header['EXTNAME'] = 'SPECTRUM', 'name of this binary table extension'
 
        #header['BACKFILE'] = backfile, 'Background FITS file'
        #header['CORRFILE'] = corrfile, 'Correlation FITS file'
        #header['RESPFILE'] = respfile, 'Redistribution matrix file (RMF)'
        #header['ANCRFILE'] = ancrfile, 'Ancillary response file (ARF)'
        
        header['HDUCLASS'] = 'OGIP', 'Format conforms to OGIP/GSFC spectral standards'
        header['HDUCLAS1'] = 'SPECTRUM', 'Extension contains a spectrum'
        header['HDUVERS '] = '1.2.1', 'Version number of the format'

        header['CHANTYPE'] = 'PHA', 'Channels assigned by detector electronics'
        header['DETCHANS'] = self.energy.nbins, 'Total number of detector channels available'
        header['TLMIN1'] = self.energy[0].value, 'Lowest Legal channel number'
        header['TLMAX1'] = self.energy[-1].value, 'Highest Legal channel number'

        header['XFLT0001'] = 'none', 'XSPEC selection filter description'

        header['HDUCLAS2'] = 'NET', 'Extension contains a bkgr substr. spec.'
        header['HDUCLAS3'] = 'COUNT', 'Extension contains counts'
        header['HDUCLAS4'] = 'TYPE:I', 'Single PHA file contained'
        header['HDUVERS1'] = '1.2.1', 'Obsolete - included for backwards compatibility'
        
        header['SYS_ERR'] = 0, 'No systematic error was specified'
        
        header['GROUPING'] = 0, 'No grouping data has been specified'
        
        header['QUALITY '] = 0, 'No data quality information specified'
        
        header['AREASCAL'] = 1., 'Nominal effective area'
        header['BACKSCAL'] = 1., 'Background scale factor'
        header['CORRSCAL'] = 0., 'Correlation scale factor'
        
        header['FILENAME'] = 'several', 'Spectrum was produced from more than one file'
        header['ORIGIN'] = 'dummy', 'origin of fits file'
        header['DATE'] = datetime.datetime.today().strftime('%Y-%m-%d'), 'FITS file creation date (yyyy-mm-dd)'
        header['PHAVERSN'] = '1992a', 'OGIP memo number for file format'
        
        return hdu

    def plot(self, ax=None, filename=None, **kwargs):
        """
        Plot effective area vs. energy.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        plt.plot(self.energy, self.counts, 'o' , **kwargs)
        plt.xlabel('Energy [{0}]'.format(self.energy.unit))
        plt.ylabel('Counts')
        if filename is not None:
            plt.savefig(filename)
            log.info('Wrote {0}'.format(filename))

        return ax


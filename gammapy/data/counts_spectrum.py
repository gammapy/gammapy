# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)

import numpy as np
from astropy import log
from ..spectrum.energy import Energy, EnergyBounds
import datetime
from astropy.io import fits
from gammapy.data import EventList, EventListDataset

__all__ = ['CountsSpectrum']


class CountsSpectrum(object):
    """Counts spectrum dataset

    Parameters
    ----------

    counts : `~numpy.array`, list
        Counts
    energy : `~gammapy.spectrum.Energy`, `~gammapy.spectrum.EnergyBounds`.
        Energy axis
    livetime : float
        Livetime of the dataset

    Examples
    --------

    .. code-block:: python

        from gammapy.spectrum import Energy, EnergyBounds
        from gammapy.data import CountsSpectrum
        ebounds = EnergyBounds.equal_log_spacing(1,10,10,'TeV') 
        counts = [6,3,8,4,9,5,9,5,5,1]
        spec = CountsSpectrum(counts, ebounds)
        hdu = spec.to_fits() 
    """

    def __init__(self, counts, energy, livetime=None, backscal=1):

        if np.asarray(counts).size != energy.nbins:
            raise ValueError("Dimension of {0} and {1} do not"
                             " match".format(counts, energy))

        if not isinstance(energy, Energy):
            raise ValueError("energy must be an Energy object")

        self.counts = np.asarray(counts)

        if isinstance(energy, EnergyBounds):
            self.energy = energy.log_centers
        else:
            self.energy = energy

        self._livetime = livetime
        self._backscal = backscal
        self.channels = np.arange(1, self.energy.nbins + 1, 1)

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

    @property
    def backscal(self):
        """Area scaling factor
        """
        return self._backscal

    @backscal.setter
    def backscal(self, value):
        self._backscal = value

    @classmethod
    def from_fits(cls, hdu):
        """Read SPECTRUM fits extension from pha file (`CountsSpectrum`).

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            ``SPECTRUM`` extensions.
        """
        pass

    @classmethod
    def from_eventlist(cls, event_list, bins):
        """Create CountsSpectrum from fits 'EVENTS' extension (`CountsSpectrum`).

        Subsets of the event list should be chosen via the appropriate methods
        in `~gammapy.data.EventList`.

        Parameters
        ----------
        event_list : `~astropy.io.fits.BinTableHDU, `gammapy.data.EventListDataSet`,
                     `gammapy.data.EventList`, str (filename)
        bins : `gammapy.spectrum.energy.EnergyBounds`
            Energy bin edges
        """

        if isinstance(event_list, fits.BinTableHDU):
            event_list = EventList.read(event_list)
        elif isinstance(event_list, EventListDataset):
            event_list = event_list.event_list
        elif isinstance(event_list, str):
            event_list = EventList.read(event_list, hdu='EVENTS')

        energy = Energy(event_list.energy).to(bins.unit)
        val, dummy = np.histogram(energy, bins.value)
        livetime = event_list.observation_live_time_duration

        return cls(val, bins, livetime)

    def write(self, filename, bkg=None, corr=None, rmf=None, arf=None,
              *args, **kwargs):
        """Write PHA to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.

        Parameters
        ----------
        arf : str
            ARF file to write into FITS header
        rmf : str
            RMF file to write into FITS header
        corr : str
            CORR file to write into FITS header
        bkg : str
            BKG file to write into FITS header
        filename : str
            File to be written
        """
        self.to_fits(bkg=bkg, corr=corr, rmf=rmf, arf=arf).writeto(
            filename, *args, **kwargs)

    def to_fits(self, bkg=None, corr=None, rmf=None, arf=None):
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
        col1 = fits.Column(name="CHANNEL", array=self.channels,
                           format='I')
        col2 = fits.Column(name="COUNTS", array=self.counts,
                           format='J', unit='count')

        cols = fits.ColDefs([col1, col2])

        hdu = fits.BinTableHDU.from_columns(cols)
        header = hdu.header

        # TODO: option to store meta info in the class
        header['EXTNAME'] = 'SPECTRUM', 'name of this binary table extension'
        header['TELESCOP'] = 'DUMMY', 'Telescope (mission) name'
        header['INSTRUME'] = 'DUMMY', 'Instrument name'
        header['FILTER'] = 'NONE', 'Instrument filter in use'
        header['EXPOSURE'] = self.livetime.to('second').value, 'Exposure time'

        header['BACKFILE'] = bkg, 'Background FITS file'
        header['CORRFILE'] = corr, 'Correlation FITS file'
        header['RESPFILE'] = rmf, 'Redistribution matrix file (RMF)'
        header['ANCRFILE'] = arf, 'Ancillary response file (ARF)'

        header['HDUCLASS'] = 'OGIP', 'Format conforms to OGIP/GSFC spectral standards'
        header['HDUCLAS1'] = 'SPECTRUM', 'Extension contains a spectrum'
        header['HDUVERS '] = '1.2.1', 'Version number of the format'

        header['CHANTYPE'] = 'PHA', 'Channels assigned by detector electronics'
        header['DETCHANS'] = self.energy.nbins, 'Total number of detector channels available'
        header['TLMIN1'] = 1, 'Lowest Legal channel number'
        header['TLMAX1'] = self.energy.nbins, 'Highest Legal channel number'

        header['XFLT0001'] = 'none', 'XSPEC selection filter description'

        header['HDUCLAS2'] = 'NET', 'Extension contains a bkgr substr. spec.'
        header['HDUCLAS3'] = 'COUNT', 'Extension contains counts'
        header['HDUCLAS4'] = 'TYPE:I', 'Single PHA file contained'
        header['HDUVERS1'] = '1.2.1', 'Obsolete - included for backwards compatibility'

        header['POISSERR'] = True, 'Are Poisson Distribution errors assumed'
        header['STAT_ERR'] = 0, 'No statisitcal error was specified'
        header['SYS_ERR'] = 0, 'No systematic error was specified'
        header['GROUPING'] = 0, 'No grouping data has been specified'

        header['QUALITY '] = 0, 'No data quality information specified'

        header['AREASCAL'] = 1., 'Nominal effective area'
        header['BACKSCAL'] = self.backscal, 'Background scale factor'
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

        plt.plot(self.energy, self.counts, 'o', **kwargs)
        plt.xlabel('Energy [{0}]'.format(self.energy.unit))
        plt.ylabel('Counts')
        if filename is not None:
            plt.savefig(filename)
            log.info('Wrote {0}'.format(filename))

        return ax

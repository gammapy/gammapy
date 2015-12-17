# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)

import numpy as np
from astropy import log

from gammapy.utils.scripts import make_path
from ..utils.energy import Energy, EnergyBounds
import datetime
from astropy.io import fits
from astropy.units import Quantity
from gammapy.data import EventList, EventListDataset

__all__ = ['CountsSpectrum']


class CountsSpectrum(object):
    """Counts spectrum dataset

    Parameters
    ----------

    counts : `~numpy.array`, list
        Counts
    energy : `~gammapy.utils.energy.EnergyBounds`
        Energy axis
    livetime : `~astropy.units.Quantiy`
        Livetime of the dataset

    Examples
    --------

    .. code-block:: python

        from gammapy.utils.energy import Energy, EnergyBounds
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

        self.counts = np.asarray(counts)

        self.energy_bounds = EnergyBounds(energy)

        self._livetime = Quantity(0, 's') if livetime is None else livetime
        self._backscal = backscal
        self.channels = np.arange(1, self.energy_bounds.nbins + 1, 1)

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

    def __add__(self, other):
        """Add two counts spectra and returns new instance

        The two spectra need to have the same binning
        """
        if (self.energy_bounds != other.energy_bounds).all():
            raise ValueError("Cannot add counts spectra with different binning")
        counts = self.counts + other.counts
        livetime = self.livetime + other.livetime
        return CountsSpectrum(counts, self.energy_bounds, livetime=livetime)

    @classmethod
    def read(cls, phafile, rmffile=None):
        """Read PHA fits file

        The energy binning is not contained in the PHA standard. Therefore is
        is inferred from the corresponding RMF EBOUNDS extension.

        Parameters
        ----------
        phafile : str
            PHA file with ``SPECTRUM`` extension
        rmffile : str
            RMF file with ``EBOUNDS`` extennsion, optional
        """
        phafile = make_path(phafile)
        spectrum = fits.open(str(phafile))['SPECTRUM']
        counts = [val[1] for val in spectrum.data]
        if rmffile is None:
            try:
                rmffile = spectrum.header['RESPFILE']
            except KeyError:
                raise ValueError('RMF file not set in PHA header. '
                                 'Please provide RMF file for energy binning')
        rmffile = make_path(rmffile)
        ebounds = fits.open(str(rmffile))['EBOUNDS']
        bins = EnergyBounds.from_ebounds(ebounds)
        livetime = Quantity(0, 's')
        return cls(counts, bins, livetime=livetime)

    @classmethod
    def from_eventlist(cls, event_list, bins):
        """Create CountsSpectrum from fits 'EVENTS' extension (`CountsSpectrum`).

        Subsets of the event list should be chosen via the appropriate methods
        in `~gammapy.data.EventList`.

        Parameters
        ----------
        event_list : `~astropy.io.fits.BinTableHDU, `gammapy.data.EventListDataSet`,
                     `gammapy.data.EventList`, str (filename)
        bins : `gammapy.utils.energy.EnergyBounds`
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
              offset=None, muon_eff=None, zenith=None, on_region=None,
              *args, **kwargs):
        """Write PHA to FITS file.

        Calls `gammapy.spectrum.CountsSpectrum.to_fits` and
        `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits(bkg=bkg, corr=corr, rmf=rmf, arf=arf, offset=offset,
                     muon_eff=muon_eff, zenith=zenith,
                     on_region=on_region).writeto(filename, *args, **kwargs)

    def to_fits(self, bkg=None, corr=None, rmf=None, arf=None, offset=None,
                muon_eff=None, zenith=None, on_region=None):
        """Convert to FITS format

        This can be used to write a :ref:`gadf:ogip-pha`

        Parameters
        ----------
        bkg : str
            :ref:`gadf:ogip-bkg` containing the corresponding background spectrum
        corr : str
            name of the corresponding correction file
        rmf : str
            :ref:`gadf:ogip-rmf` containing the corresponding energy resolution
        arf : str
            :ref:`gadf:ogip-arf` containing the corresponding effective area
        offset : `~astropy.coordinates.Angle`
            Angular distance between target and pointing position
        muon_eff : float
            Muon efficiency
        zenith : `~astropy.coordinates.Angle`
            Zenith angle
        on_region : `~gammapy.region.SkyCircleRegion`
            Region used to extract the spectrum

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
        header['DETCHANS'] = self.channels.size, 'Total number of detector channels available'
        header['TLMIN1'] = 1, 'Lowest Legal channel number'
        header['TLMAX1'] = self.channels[-1], 'Highest Legal channel number'

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

        header['AREASCAL'] = 1., 'Area scaling factor'
        header['BACKSCAL'] = self.backscal, 'Background scale factor'
        header['CORRSCAL'] = 0., 'Correlation scale factor'

        header['FILENAME'] = 'several', 'Spectrum was produced from more than one file'
        header['ORIGIN'] = 'dummy', 'origin of fits file'
        header['DATE'] = datetime.datetime.today().strftime('%Y-%m-%d'), 'FITS file creation date (yyyy-mm-dd)'
        header['PHAVERSN'] = '1992a', 'OGIP memo number for file format'

        if offset is not None:
            header['OFFSET'] = offset.to('deg').value, 'Target offset from pointing position'
        if muon_eff is not None:
            header['MUONEFF'] = muon_eff, 'Muon efficiency'
        if zenith is not None:
            header['ZENITH'] = zenith.to('deg').value, 'Zenith angle'
        if on_region is not None:
            header['RA-OBJ'] = on_region.pos.icrs.ra.value, 'Right ascension of the target'
            header['DEC-OBJ'] = on_region.pos.icrs.dec.value , 'Declination of the target'
            header['ON-RAD'] = on_region.radius.to('deg').value, 'Radius of the circular spectral extraction region'

        return hdu

    def plot(self, ax=None, filename=None, weight=1, **kwargs):
        """
        Plot counts vector

        kwargs are forwarded to matplotlib.pyplot.hist

        Parameters
        ----------
        ax : `~matplotlib.axis` (optional)
            Axis instance to be used for the plot
        filename : str (optional)
            File to save the plot to
        weight : float
            Weighting factor for the counts

        Returns
        -------
        ax: `~matplotlib.axis`
            Axis instance used for the plot
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        w = self.counts * weight
        plt.hist(self.energy_bounds.log_centers, bins=self.energy_bounds,
                 weights=w, **kwargs)
        plt.xlabel('Energy [{0}]'.format(self.energy_bounds.unit))
        plt.ylabel('Counts')
        if filename is not None:
            plt.savefig(filename)
            log.info('Wrote {0}'.format(filename))

        return ax

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)

import numpy as np
from astropy.coordinates import Angle

from ..extern.bunch import Bunch
from ..utils.array import array_stats_str
from ..utils.scripts import make_path
from ..utils.energy import Energy, EnergyBounds
import datetime
from astropy.io import fits
from astropy.units import Quantity
from ..data import EventList, EventListDataset
from ..extern.pathlib import Path

__all__ = ['CountsSpectrum']


class CountsSpectrum(object):
    """Counts spectrum dataset

    Parameters
    ----------
    counts : `~numpy.array`, list
        Counts
    energy : `~gammapy.utils.energy.EnergyBounds`
        Energy axis

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

    def __init__(self, counts, energy, meta=None):

        if np.asarray(counts).size != energy.nbins:
            raise ValueError("Dimension of {0} and {1} do not"
                             " match".format(counts, energy))

        self.counts = np.asarray(counts)
        self.energy_bounds = EnergyBounds(energy)

        self.meta = Bunch(meta) if meta is not None else Bunch()
        # This is needed to write valid OGIP files
        self.meta.setdefault('livetime', Quantity(0, 's'))
        self.meta.setdefault('backscal', 1)
        self.channels = np.arange(1, self.energy_bounds.nbins + 1, 1)

    @classmethod
    def read_pha(cls, phafile, rmffile=None):
        """Read PHA fits file

        The energy binning is not contained in the PHA standard. Therefore is
        is inferred from the corresponding RMF EBOUNDS extension.
        Todo: Should the energy binning be in the PHA file?

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
        header = spectrum.header
        if rmffile is None:
            val = header['RESPFILE']
            if val == '':
                raise ValueError('RMF file not set in PHA header. '
                                 'Please provide RMF file for energy binning')
            parts = phafile.parts[:-1]
            rmffile = Path.cwd()
            for part in parts:
                rmffile = rmffile.joinpath(part)
            rmffile = rmffile.joinpath(val)

        rmffile = make_path(rmffile)
        ebounds = fits.open(str(rmffile))['EBOUNDS']
        bins = EnergyBounds.from_ebounds(ebounds)
        m = dict()

        # Todo: think about better way to handle this
        m.update(backscal=header['BACKSCAL'])
        m.update(obs_id=header['OBS_ID'])
        m.update(livetime=Quantity(header['EXPOSURE'], 's'))
        m.update(rmf=header['RESPFILE'])
        m.update(arf=header['ANCRFILE'])
        m.update(bkg=header['BACKFILE'])
        if 'LO_THRES' in header.keys():
            rng = EnergyBounds([header['LO_THRES'], header['HI_THRES']], 'TeV')
            m.update(safe_energy_range=rng)

        return cls(counts, bins, meta=m)

    @classmethod
    def read_bkg(cls, bkgfile, rmffile):
        """Read BKG fits file

        The energy binning is not contained in the PHA standard. Therefore is
        is inferred from the corresponding RMF EBOUNDS extension.
        Todo: Should the energy binning be in the BKG file?

        Parameters
        ----------
        phafile : str
            PHA file with ``SPECTRUM`` extension
        rmffile : str
            RMF file with ``EBOUNDS`` extennsion
        """
        bkgfile = make_path(bkgfile)
        spectrum = fits.open(str(bkgfile))['SPECTRUM']
        counts = [val[1] for val in spectrum.data]
        rmffile = make_path(rmffile)
        ebounds = fits.open(str(rmffile))['EBOUNDS']
        bins = EnergyBounds.from_ebounds(ebounds)
        header = spectrum.header
        m = dict()
        m.update(backscal=header['BACKSCAL'])
        m.update(livetime=Quantity(header['EXPOSURE'], 's'))

        return cls(counts, bins, meta=m)

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
        meta = dict(livetime=livetime)

        return cls(val, bins)

    @classmethod
    def get_npred(cls, fit, obs):
        """Get N_pred vector from spectral fit

        Parameters
        ----------
        fit : SpectrumFitResult
            Fitted spectrum
        obs : SpectrumObservationList
            Spectrum observation holding the irfs
        """

        m = fit.to_sherpa_model()

        # Get differential flux at true energy log bin center
        ebounds = obs.effective_area.ebounds
        x = ebounds.log_centers.to('keV')
        diff_flux = Quantity(m(x), 'cm-2 s-1 keV-1')

        # Multiply with bin width = integration
        int_flux = (diff_flux * ebounds.bands).decompose()

        # Apply ARF and RMF to get n_pred
        temp = int_flux * obs.meta.livetime * obs.effective_area.effective_area
        counts = obs.energy_dispersion.pdf_matrix.transpose().dot(temp)

        e_reco = obs.energy_dispersion.reco_energy
        return cls(counts.decompose(), e_reco)

    @property
    def total_counts(self):
        """Total number of counts
        """
        return self.counts.sum()

    def info(self):
        """Info string
        """
        ss = "\nSummary CountsSpectrum info\n"
        ss += "----------------\n"
        # Summarise data members
        ss += array_stats_str(self.counts, 'counts')
        ss += array_stats_str(self.energy_bounds.to('TeV'), 'energy')
        ss += 'Total Counts: {}'.format(self.total_counts)

        return ss

    def __add__(self, other):
        """Add two counts spectra and returns new instance
        The two spectra need to have the same binning
        """
        if (self.energy_bounds != other.energy_bounds).all():
            raise ValueError("Cannot add counts spectra with different binning")
        counts = self.counts + other.counts
        meta = dict(livetime=self.meta.livetime + other.meta.livetime)
        return CountsSpectrum(counts, self.energy_bounds, meta=meta)

    def __mul__(self, other):
        """Scale counts by a factor"""
        temp = self.counts * other
        meta = dict(livetime=self.meta.livetime)
        return CountsSpectrum(temp, self.energy_bounds, meta=meta)

    def write(self, filename, bkg=None, corr=None, rmf=None, arf=None,
              *args, **kwargs):
        """Write PHA to FITS file.

        Calls `gammapy.spectrum.CountsSpectrum.to_fits` and
        `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits(bkg=bkg, corr=corr, rmf=rmf, arf=arf).writeto(
            filename, *args, **kwargs)

    def to_fits(self, bkg=None, corr=None, rmf=None, arf=None):
        """Convert to FITS format

        This can be used to write a :ref:`gadf:ogip-pha`. Meta info is written
        in the fits header.

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
        header['EXPOSURE'] = self.meta.livetime.to('second').value, 'Exposure time'

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
        header['BACKSCAL'] = self.meta.backscal, 'Background scale factor'
        header['CORRSCAL'] = 0., 'Correlation scale factor'

        header['FILENAME'] = 'several', 'Spectrum was produced from more than one file'
        header['ORIGIN'] = 'dummy', 'origin of fits file'
        header['DATE'] = datetime.datetime.today().strftime('%Y-%m-%d'), 'FITS file creation date (yyyy-mm-dd)'
        header['PHAVERSN'] = '1992a', 'OGIP memo number for file format'

        # Todo: think about better way to handel this
        val = self.meta.keys()
        if 'obs_id' in val:
            header['OBS_ID'] = self.meta.obs_id
        if 'safe_energy_range' in val:
            header['HI_THRES'] = self.meta.safe_energy_range[1].to('TeV').value, 'Low energy threshold [TeV] for spectral fit'
            header['LO_THRES'] = self.meta.safe_energy_range[0].to('TeV').value, 'High energy threshold [TeV] for spectral fit'

        return hdu

    def plot(self, ax=None, weight=1, energy_unit='TeV', **kwargs):
        """
        Plot counts vector

        kwargs are forwarded to matplotlib.pyplot.hist

        Parameters
        ----------
        ax : `~matplotlib.axis` (optional)
            Axis instance to be used for the plot
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
        e = self.energy_bounds.to(energy_unit)
        plt.hist(e.log_centers, bins=e, weights=w, **kwargs)
        plt.xlabel('Energy [{0}]'.format(energy_unit))
        plt.ylabel('Counts')

        return ax

    def peek(self, figsize=(5, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt
        ax = plt.figure(figsize=figsize)
        self.plot(ax=ax)
        plt.xscale('log')
        plt.show()

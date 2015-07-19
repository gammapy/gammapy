# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import datetime
import numpy as np
from astropy.units import Quantity
from astropy.io import fits
from ..utils.array import array_stats_str

__all__ = ['LogEnergyAxis',
           'energy_bounds_equal_log_spacing',
           'EnergyBinning',
           'np_to_pha',
           ]


def find_log_centers(bin_edges):
    """Compute energy bin log centers.

    Bin center = sqrt(low_edge * high_edge)
    """
    return np.sqrt(bin_edges[:-1] * bin_edges[1:])


def find_log_edges(bin_centers):
    """Compute equally log-spaced energy bin edges
    """

    bin_edges = np.sqrt(bin_centers[:-1] * bin_centers[1:])
    first = bin_centers[0] * bin_centers[0] / bin_edges[0]
    last = bin_centers[-1] * bin_centers[-1] / bin_edges[-1]
    bin_edges = np.append(first, bin_edges)
    bin_edges = np.append(bin_edges, last)

    return bin_edges


class EnergyBinning(object):

    """Class to handle energy dimension for e.g. counts spectra, ARF tables
    and so on. The idea is to handel everything related to energy in this
    class. It should probably be move somewhere else (not spectrum.utils)
    rather datasets.energyhandler or similar. All other classes should then
    takes this class as argument.

    TODO:
    - document once we agreed on a mechanism
    - implement FITS I/O

 
    Parameters
    ----------
    bin_edges : `~astropy.units.Quantity`
    Energy bin edges
    bin_centers: `~astropy.units.Quantity`
    Energy bin centers


    Examples
    --------

    Create EnergyBinning by giving the bin centers and write FITS ENERGIES extension:
    
    .. code-block:: python
    
       from gammapy.spectrum.utils import *  
       from astropy.units import Quantity
       from astropy.io import fits
       bin_centers = Quantity([0.1,1,1.5,2,2.3], 'TeV')
       binning = EnergyBinning(bin_centers=bin_centers)          
       prihdr = fits.Header()
       prihdu = fits.PrimaryHDU(header=prihdr)
       tbhdu = binning.to_fits('ENERGIES')
       hdulist = fits.HDUList([prihdu, tbhdu])
       hdulist.writeto('test.fits')

    Create EnergyBinning by giving the bin edges and place centers at log 

    .. code-block:: python

       from gammapy.spectrum.utils import *  
       from astropy.units import Quantity
       import numpy as np
       bin_edges = np.logspace(-1,1,20)
       bin_edges = Quantity(bin_edges, 'TeV')
       binning = EnergyBinning.from_edges(bin_edges, 'log')

    """

    def __init__(self, bin_edges=None, bin_centers=None):

        if not isinstance(bin_edges, Quantity) and not isinstance(bin_centers, Quantity):
            raise ValueError("Energies must be Quantity objects")

        self._bin_edges = bin_edges
        self._bin_centers = bin_centers

        if bin_centers is not None:
            self.nbins = len(self.bin_centers)
        elif bin_edges is not None:
            self.nbins = len(self.bin_edges) - 1
        else:
            self.nbins = 0

    @staticmethod
    def from_edges(bin_edges, centers='log'):
        """Create EnergyBinning by given the bin edges and a method how to place the bin centers.

        Options for placing the bin centers are:

        - 'log' : Place bin centers at the log center of the bin (default)
        - to be implemented

        Parameters
        ----------
        bin_edges : `~astropy.units.Quantity`
        Energy bin edges
        centeres : str
        Method how to place the energy bin centers

        Returns
        -------
        Energy Binning
        """
        if centers == 'log':
            bin_centers = find_log_centers(bin_edges)
        else:
            raise ValueError("Method for placing bin centers not implemented")

        return EnergyBinning(bin_edges, bin_centers)

    @staticmethod
    def from_centers(bin_centers):
        """Create EnergyBinning by given the bin centers

        The difference between this method and the contrsuctor is that a check for equal log spacing is done. If the given bin centers are equally spaced in log, the bin edges are computed  (I don't know if thats really necessary, it doesn't hurt for sure)
        Parameters
        ----------
        bin_centers : `~astropy.units.Quantity`
        Energy bin centers

        Returns
        -------
        Energy Binning
        """

        # TODO: test for equal log spacing
        bin_edges = find_log_edges(bin_centers)

        return EnergyBinning(bin_edges, bin_centers)

    @staticmethod
    def equal_log_spacing(emin, emax, nbins):
        """Create EnergyBinning with equal log-spacing by giving the outer bin edges

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
        Lower bound of lowest energy bin
        emax : `~astropy.units.Quantity`
        Upper bound of highest energy bin
        bins : int
        Number of bins

        Returns
        -------
        Energy Binning
        """

        if not isinstance(emin, Quantity) or not isinstance(emax, Quantity):
            raise ValueError("Energies must be Quantity objects.")

        emin = emin.to('TeV')
        emax = emax.to('TeV')

        x_min, x_max = np.log10([emin.value, emax.value])
        bin_edges = np.logspace(x_min, x_max, nbins + 1)

        bin_edges = Quantity(bin_edges, emin.unit)

        return EnergyBinning.from_edges(bin_edges, 'log')

    @staticmethod
    def from_fits(hdulist):
        """Read energy axis from hdulist

        Support all formats here
        """

        try:
            bin_centers = Quantity(hdulist['ENERGIES'].data['Energy'], 'MeV')
            return EnergyBinning.from_centers(bin_centers)
        except KeyError:
            pass

        try:
            energy_lo = hdulist['SPECRESP'].data['ENERG_LO']
            energy_hi = hdulist['SPECRESP'].data['ENERG_HI']
            bin_edges = Quantity(energy_lo + energy_hi[-1],
                                 hdulist['SPECRESP'].header['TUNIT1'])
            return EnergyBinning.from_edges(bin_edges, centers='log')
        except KeyError:
            pass

        raise ValueError("The hdulist does not contain a valid energy extension")

    def to_fits(self, name, **kwargs):
        """Write energy axis to hdulist
        """

        if name == 'ENERGIES':
            #this sould go to a separate function in the end I guess
            #TODO write header, and so on
            col1 = fits.Column(name='Energy', format='D', array=self.bin_centers)
            cols = fits.ColDefs([col1])
            return fits.BinTableHDU.from_columns(cols)

    @property
    def bin_centers(self):
        """
        Centers of the energy bins
        """

        if self._bin_centers is not None:
            return self._bin_centers
        else:
            raise ValueError("Bin centers not defined for this energy axis")

    @property
    def bin_edges(self):
        """
        Edges of the energy bins
        """

        if self._bin_edges is not None:
            return self._bin_edges
        else:
            raise ValueError("Bin edges not defined for this energy axis")

    def info(self):
        s = '\nEnergy bins\n'
        s += "------------\n"
        s += 'Nbins: {0}\n'.format(self.nbins)
        try:
            s += array_stats_str(self.bin_edges, 'Energy bin edges')
        except ValueError:
            s += 'Bin edges not defined\n'
        try:
            s += array_stats_str(self.bin_centers, 'Energy bin centers')
        except ValueError:
            s += 'Bin centers not defined\n'
        return s


class LogEnergyAxis(object):

    """Log10 energy axis.

    Defines a transformation between:

    * ``energy = 10 ** x``
    * ``x = log10(energy)``
    * ``pix`` in the range [0, ..., len(x)] via linear interpolation of the ``x`` array,
      e.g. ``pix=0`` corresponds to ``x[0]``
      and ``pix=0.3`` is ``0.5 * (0.3 * x[0] + 0.7 * x[1])``

    .. note::
        The `specutils.Spectrum1DLookupWCS <http://specutils.readthedocs.org/en/latest/api/specutils.wcs.specwcs.Spectrum1DLookupWCS.html>`__
        class is similar (only that it doesn't include the ``log`` transformation and the API is different.
        Also see this Astropy feature request: https://github.com/astropy/astropy/issues/2362

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy array
    """

    def __init__(self, energy):

        self.energy = energy
        self.x = np.log10(energy.value)
        self.pix = np.arange(len(self.x))

    def world2pix(self, energy):
        """TODO: document.
        """
        # Convert `energy` to `x = log10(energy)`
        x = np.log10(energy.to(self.energy.unit).value)

        # Interpolate in `x`
        pix = np.interp(x, self.x, self.pix)

        return pix

    def pix2world(self, pix):
        """TODO: document.
        """
        # Interpolate in `x = log10(energy)`
        x = np.interp(pix, self.pix, self.x)

        # Convert `x` to `energy`
        energy = Quantity(10 ** x, self.energy.unit)

        return energy

    def closest_point(self, energy):
        """TODO: document
        """
        x = np.log10(energy.value)
        # TODO: I'm not sure which is faster / better here?
        index = np.argmin(np.abs(self.x - x))
        # np.searchsorted(self.x, x)
        return index

    def bin_edges(self, energy):
        """TODO: document.

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        try:
            pix = np.where(energy >= self.energy)[0][-1]
        except ValueError:
            # Loop over es by hand
            pix1 = np.empty_like(energy, dtype=int)
            for ii in range(energy.size):
                # print ii, e[ii], np.where(e[ii] >= self.e)
                pix1[ii] = np.where(energy[ii] >= self.energy)[0][-1]
        pix2 = pix1 + 1
        energy1 = self.energy[pix1]
        energy2 = self.energy[pix2]

        return pix1, pix2, energy1, energy2


# MOVE TO DATA.COUNTSPECTRUM

def np_to_pha(channel, counts, exposure, dstart, dstop,
              dbase=None, stat_err=None, quality=None, syserr=None,
              obj_ra=0., obj_dec=0., obj_name='DUMMY', creator='DUMMY',
              version='v0.0.0', telescope='DUMMY', instrument='DUMMY', filter='NONE',
              backfile='none', corrfile='none', respfile='none', ancrfile='none'):

    """Create PHA FITS table extension from numpy arrays.

    Parameters
    ----------
    dat : numpy 1D array float
        Binned spectral data [counts]
    dat_err : numpy 1D array float
        Statistical errors associated with dat [counts]
    chan : numpu 1D array int
        Corresponding channel numbers for dat
    exposure : float
        Exposure [s]
    dstart : astropy.time.Time
        Observation start time.
    dstop : astropy.time.Time
        Observation stop time.
    dbase : astropy.time.Time
        Base date used for TSTART/TSTOP.
    quality : numpy 1D array integer
        Quality flags for the channels (optional)
    syserr : numpy 1D array float
        Fractional systematic error for the channel (optional)
    obj_ra/obj_dec : float
        Object RA/DEC J2000 [deg]

    Returns
    -------
    pha : `~astropy.io.fits.BinTableHDU`
        PHA FITS HDU

    Notes
    -----
    For more info on the PHA FITS file format see:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/summary/ogip_92_007_summary.html
    """
    # Create PHA FITS table extension from data
    cols = [fits.Column(name='CHANNEL',
                        format='I',
                        array=channel,
                        unit='channel'),
            fits.Column(name='COUNTS',
                        format='1E',
                        array=counts,
                        unit='count')
            ]

    if stat_err is not None:
        cols.append(fits.Column(name='STAT_ERR',
                                format='1E',
                                array=stat_err,
                                unit='count'))

    if syserr is not None:
        cols.append(fits.Column(name='SYS_ERR',
                                format='E',
                                array=syserr))

    if quality is not None:
        cols.append(fits.Column(name='QUALITY',
                                format='I',
                                array=quality))

    hdu = fits.new_table(cols)
    header = hdu.header

    header['EXTNAME'] = 'SPECTRUM', 'name of this binary table extension'
    header['TELESCOP'] = telescope, 'Telescope (mission) name'
    header['INSTRUME'] = instrument, 'Instrument name'
    header['FILTER'] = filter, 'Instrument filter in use'
    header['EXPOSURE'] = exposure, 'Exposure time'

    header['BACKFILE'] = backfile, 'Background FITS file'
    header['CORRFILE'] = corrfile, 'Correlation FITS file'
    header['RESPFILE'] = respfile, 'Redistribution matrix file (RMF)'
    header['ANCRFILE'] = ancrfile, 'Ancillary response file (ARF)'

    header['HDUCLASS'] = 'OGIP', 'Format conforms to OGIP/GSFC spectral standards'
    header['HDUCLAS1'] = 'SPECTRUM', 'Extension contains a spectrum'
    header['HDUVERS '] = '1.2.1', 'Version number of the format'

    poisserr = False
    if stat_err is None:
        poisserr = True
    header['POISSERR'] = poisserr, 'Are Poisson Distribution errors assumed'

    header['CHANTYPE'] = 'PHA', 'Channels assigned by detector electronics'
    header['DETCHANS'] = len(channel), 'Total number of detector channels available'
    header['TLMIN1'] = channel[0], 'Lowest Legal channel number'
    header['TLMAX1'] = channel[-1], 'Highest Legal channel number'

    header['XFLT0001'] = 'none', 'XSPEC selection filter description'
    header['OBJECT'] = obj_name, 'OBJECT from the FIRST input file'
    header['RA-OBJ'] = obj_ra, 'RA of First input object'
    header['DEC-OBJ'] = obj_dec, 'DEC of First input object'
    header['EQUINOX'] = 2000.00, 'Equinox of the FIRST object'
    header['RADECSYS'] = 'FK5', 'Co-ordinate frame used for equinox'
    header['DATE-OBS'] = dstart.datetime.strftime('%Y-%m-%d'), 'EARLIEST observation date of files'
    header['TIME-OBS'] = dstart.datetime.strftime('%H:%M:%S'), 'EARLIEST time of all input files'
    header['DATE-END'] = dstop.datetime.strftime('%Y-%m-%d'), 'LATEST observation date of files'
    header['TIME-END'] = dstop.datetime.strftime('%H:%M:%S'), 'LATEST time of all input files'

    header['CREATOR'] = '{0} {1}'.format(creator, version), 'Program name that produced this file'

    header['HDUCLAS2'] = 'NET', 'Extension contains a bkgr substr. spec.'
    header['HDUCLAS3'] = 'COUNT', 'Extension contains counts'
    header['HDUCLAS4'] = 'TYPE:I', 'Single PHA file contained'
    header['HDUVERS1'] = '1.2.1', 'Obsolete - included for backwards compatibility'

    if syserr is None:
        header['SYS_ERR'] = 0, 'No systematic error was specified'

    header['GROUPING'] = 0, 'No grouping data has been specified'

    if quality is None:
        header['QUALITY '] = 0, 'No data quality information specified'

    header['AREASCAL'] = 1., 'Nominal effective area'
    header['BACKSCAL'] = 1., 'Background scale factor'
    header['CORRSCAL'] = 0., 'Correlation scale factor'

    header['FILENAME'] = 'several', 'Spectrum was produced from more than one file'
    header['ORIGIN'] = 'dummy', 'origin of fits file'
    header['DATE'] = datetime.datetime.today().strftime('%Y-%m-%d'), 'FITS file creation date (yyyy-mm-dd)'
    header['PHAVERSN'] = '1992a', 'OGIP memo number for file format'

    if dbase is not None:
        header['TIMESYS'] = 'MJD', 'The time system is MJD'
        header['TIMEUNIT'] = 's', 'unit for TSTARTI/F and TSTOPI/F, TIMEZERO'
        header['MJDREF'] = dbase.mjd, 'MJD for reference time'
        header['TSTART'] = (dstart - dbase).sec, 'Observation start time [s]'
        header['TSTOP'] = (dstop - dbase).sec, 'Observation stop time [s]'

    return hdu

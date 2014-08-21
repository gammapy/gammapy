# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import datetime
import numpy as np
from astropy.units import Quantity
from astropy.io import fits

__all__ = ['LogEnergyAxis',
           'energy_bounds_equal_log_spacing',
           'energy_bin_centers_log_spacing',
           'np_to_pha',
           ]


def energy_bin_centers_log_spacing(energy_bounds):
    """Compute energy log bin centers.

    TODO: give formula here.

    Parameters
    ----------
    energy_bounds : `~astropy.units.Quantity`
        Array of energy bin edges.

    Returns
    -------
    energy_center : `~astropy.units.Quantity`
        Array of energy bin centers
    """
    e_bounds = energy_bounds.value
    e_center = np.sqrt(e_bounds[:-1] * e_bounds[1:])
    return Quantity(e_center, energy_bounds.unit)


def energy_bounds_equal_log_spacing(energy_band, bins=10):
    """Make energy bounds array with equal-log spacing.

    Parameters
    ----------
    energy_band : `~astropy.units.Quantity`
        Tuple ``(energy_min, energy_max)``
    bins : int
        Number of bins

    Returns
    -------
    energy_bounds : `~astropy.units.Quantity`
        Energy bounds array (1-dim with length ``bins + 1``).
    """
    x_min, x_max = np.log10(energy_band.value)
    energy_bounds = np.logspace(x_min, x_max, bins + 1)
    energy_bounds = Quantity(energy_bounds, energy_band.unit)

    return energy_bounds


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


def np_to_pha(channel, counts, exposure, dstart, dstop,
              dbase=None, stat_err=None, quality=None, syserr=None,
              obj_ra=0., obj_dec=0., obj_name='DUMMY', creator='DUMMY',
              version='v0.0.0', telescope='DUMMY', instrument='DUMMY', filter='NONE'):
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

    header['BACKFILE'] = 'none', 'Background FITS file'
    header['CORRFILE'] = 'none', 'Correlation FITS file'
    header['RESPFILE'] = 'none', 'Redistribution matrix file (RMF)'
    header['ANCRFILE'] = 'none', 'Ancillary response file (ARF)'

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

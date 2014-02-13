# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import datetime
import numpy as np
from astropy.io import fits

__all__ = ['EnergyAxis', 'np_to_pha']

class EnergyAxis(object):
    """Log(E) axis"""
    def __init__(self, e):
        self.e = e
        self.log_e = np.log10(e)

    def __call__(self, e):
        try:
            z1 = np.where(e >= self.e)[0][-1]
        except ValueError:
            # Loop over es by hand
            z1 = np.empty_like(e, dtype=int)
            for ii in range(e.size):
                # print ii, e[ii], np.where(e[ii] >= self.e)
                z1[ii] = np.where(e[ii] >= self.e)[0][-1]
        z2 = z1 + 1
        e1 = self.e[z1]
        e2 = self.e[z2]
        # print e1, '<=', e, '<', e2
        return z1, z2, e1, e2


def np_to_pha(channel, counts, exposure, dstart, dstop, dbase=None, stat_err=None, quality=None, syserr=None,
              obj_ra=0., obj_dec=0., obj_name='DUMMY', creator='DUMMY',
              version='v0.0.0', telescope='DUMMY', instrument='DUMMY', filter_='NONE') :
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

    if stat_err is not None :
        cols.append(fits.Column(name='STAT_ERR',
                                  format='1E',
                                  array=stat_err,
                                  unit='count'))
    
    if syserr is not None :
        cols.append(fits.Column(name='SYS_ERR',
                                  format='E',
                                  array=syserr))

    if quality is not None :
        cols.append(fits.Column(name='QUALITY',
                                  format='I',
                                  array=quality))

    hdu = fits.new_table(cols)
    header = hdu.header

    header['EXTNAME'] = 'SPECTRUM', 'name of this binary table extension'
    header['TELESCOP'] = telescope, 'Telescope (mission) name'
    header['INSTRUME'] = instrument, 'Instrument name'
    header['FILTER'] = filter_, 'Instrument filter in use'
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

    if dbase :
        header['TIMESYS'] = 'MJD', 'The time system is MJD'
        header['TIMEUNIT'] = 's', 'unit for TSTARTI/F and TSTOPI/F, TIMEZERO'
        header['MJDREF'] = dbase.mjd, 'MJD for reference time'
        header['TSTART'] = (dstart - dbase).sec, 'Observation start time [s]'
        header['TSTOP'] = (dstop - dbase).sec, 'Observation stop time [s]'

    return hdu

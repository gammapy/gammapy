# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time related utility functions."""
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

__all__ = [
    "absolute_time",
    "time_ref_from_dict",
    "time_ref_to_dict",
    "time_relative_to_ref",
]

# TODO: implement and document this properly.
# see https://github.com/gammapy/gammapy/issues/284
TIME_REF_FERMI = Time("2001-01-01T00:00:00")

#: Default epoch gammapy uses for FITS files (MJDREF)
#: 0 MJD, TT
DEFAULT_EPOCH = Time(0, format="mjd", scale="tt")


def time_to_fits(time, epoch=None, unit=u.s):
    """Convert time to fits format.

    Times are stored as duration since an epoch in FITS.


    Parameters
    ----------
    time : `~astropy.time.Time`
        time to be converted
    epoch : `astropy.time.Time`
        epoch to use for the time. The corresponding keywords must
        be stored in the same FITS header.
        If None, `DEFAULT_EPOCH` will be used.
    unit : `astropy.time.Time`
        If None, `DEFAULT_EPOCH` will be used.
        Should be stored as `TIMEUNIT` in the same FITS header.

    Returns
    -------
    time : astropy.units.Quantity
        Duration since epoch
    """
    if epoch is None:
        epoch = DEFAULT_EPOCH
    return (time - epoch).to(unit)


def time_to_fits_header(time, epoch=None, unit=u.s):
    """Convert time to fits header format.

    Times are stored as duration since an epoch in FITS.


    Parameters
    ----------
    time : `~astropy.time.Time`
        time to be converted
    epoch : `astropy.time.Time`
        epoch to use for the time. The corresponding keywords must
        be stored in the same FITS header.
        If None, `DEFAULT_EPOCH` will be used.
    unit : `astropy.time.Time`
        If None, `DEFAULT_EPOCH` will be used.
        Should be stored as `TIMEUNIT` in the same FITS header.

    Returns
    -------
    header_entry : tuple(float, string)
        a float / comment tuple with the time and unit
    """
    if epoch is None:
        epoch = DEFAULT_EPOCH
    time = time_to_fits(time)
    return time.to_value(unit), unit.to_string("fits")


def time_ref_from_dict(meta, format="mjd", scale="tt"):
    """Calculate the time reference from metadata.

    Parameters
    ----------
    meta : dict
        FITS time standard header info

    Returns
    -------
    time : `~astropy.time.Time`
        Time object with ``format='MJD'``
    """
    # Note: the float call here is to make sure we use 64-bit
    mjd = float(meta["MJDREFI"]) + float(meta["MJDREFF"])
    scale = meta.get("TIMESYS", scale).lower()
    return Time(mjd, format=format, scale=scale)


def time_ref_to_dict(time=None, scale="tt"):
    """Convert the epoch to the relevant FITS header keywords

    Parameters
    ----------
    time : `~astropy.time.Time`
        The reference epoch for storing time in FITS.

    Returns
    -------
    meta : dict
        FITS time standard header info
    """
    if time is None:
        time = DEFAULT_EPOCH
    mjd = Time(time, scale=scale).mjd
    i = np.floor(mjd).astype(np.int64)
    f = mjd - i
    return dict(MJDREFI=i, MJDREFF=f, TIMESYS=scale)


def time_relative_to_ref(time, meta):
    """Convert a time using an existing reference.

    The time reference is built as MJDREFI + MJDREFF in units of MJD.
    The time will be converted to seconds after the reference.

    Parameters
    ----------
    time : `~astropy.time.Time`
        time to be converted
    meta : dict
        dictionary with the keywords ``MJDREFI`` and ``MJDREFF``

    Returns
    -------
    time_delta : `~astropy.time.TimeDelta`
        time in seconds after the reference
    """
    time_ref = time_ref_from_dict(meta)
    return TimeDelta(time - time_ref, format="sec")


def absolute_time(time_delta, meta):
    """Convert a MET into human readable date and time.

    Parameters
    ----------
    time_delta : `~astropy.time.TimeDelta`
        time in seconds after the MET reference
    meta : dict
        dictionary with the keywords ``MJDREFI`` and ``MJDREFF``

    Returns
    -------
    time : `~astropy.time.Time`
        absolute time with ``format='ISOT'`` and ``scale='UTC'``
    """
    time = time_ref_from_dict(meta) + time_delta
    return Time(time.utc.isot)

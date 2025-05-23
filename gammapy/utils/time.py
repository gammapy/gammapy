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

TIME_KEYWORDS = ["MJDREFI", "MJDREFF", "TIMEUNIT", "TIMESYS", "TIMEREF"]

# TODO: implement and document this properly.
# see https://github.com/gammapy/gammapy/issues/284
TIME_REF_FERMI = Time("2001-01-01T00:00:00")

# Default time ref used for GTIs
TIME_REF_DEFAULT = Time("2000-01-01T00:00:00", scale="tt")

#: Default epoch gammapy uses for FITS files (MJDREF)
#: 0 MJD, TT
DEFAULT_EPOCH = Time(0, format="mjd", scale="tt")


def time_to_fits(time, epoch=None, unit=u.s):
    """Convert time to fits format.

    Times are stored as duration since an epoch in FITS.


    Parameters
    ----------
    time : `~astropy.time.Time`
        Time to be converted.
    epoch : `~astropy.time.Time`, optional
        Epoch to use for the time. The corresponding keywords must
        be stored in the same FITS header.
        Default is None, so the `DEFAULT_EPOCH` is used.
    unit : `~astropy.units.Unit`, optional
        Unit, should be stored as `TIMEUNIT` in the same FITS header.
        Default is u.s.

    Returns
    -------
    time : astropy.units.Quantity
        Duration since epoch.
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
        Time to be converted.
    epoch : `~astropy.time.Time`, optional
        Epoch to use for the time. The corresponding keywords must
        be stored in the same FITS header.
        Default is None, so `DEFAULT_EPOCH` is used.
    unit : `~astropy.units.Unit`, optional
        Unit, should be stored as `TIMEUNIT` in the same FITS header.
        Default is u.s.

    Returns
    -------
    header_entry : tuple(float, string)
        A float / comment tuple with the time and unit.
    """
    if epoch is None:
        epoch = DEFAULT_EPOCH
    time = time_to_fits(time, epoch)
    return time.to_value(unit), unit.to_string("fits")


def time_ref_from_dict(meta, format="mjd", scale="tt"):
    """Calculate the time reference from metadata.

    Parameters
    ----------
    meta : dict
        FITS time standard header information.
    format: str, optional
        Format of the `~astropy.time.Time` information. Default is 'mjd'.
    scale: str, optional
        Scale of the `~astropy.time.Time` information. Default is 'tt'.

    Returns
    -------
    time : `~astropy.time.Time`
        Time object with ``format='MJD'``.
    """
    scale = meta.get("TIMESYS", scale).lower()

    # some files seem to have MJDREFF as string, not as float
    mjdrefi = float(meta["MJDREFI"])
    mjdreff = float(meta["MJDREFF"])
    return Time(mjdrefi, mjdreff, format=format, scale=scale)


def time_ref_to_dict(time=None, scale="tt"):
    """Convert the epoch to the relevant FITS header keywords.

    Parameters
    ----------
    time : `~astropy.time.Time`, optional
        The reference epoch for storing time in FITS.
        Default is None, so 'DEFAULT_EPOCH' is used.
    scale: str, optional
        Scale of the `~astropy.time.Time` information.
        Default is "tt".

    Returns
    -------
    meta : dict
        FITS time standard header information.
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
        Time to be converted.
    meta : dict
        Dictionary with the keywords ``MJDREFI`` and ``MJDREFF``.

    Returns
    -------
    time_delta : `~astropy.time.TimeDelta`
        Time in seconds after the reference.
    """
    time_ref = time_ref_from_dict(meta)
    return TimeDelta(time - time_ref, format="sec")


def absolute_time(time_delta, meta):
    """Convert a MET into human-readable date and time.

    Parameters
    ----------
    time_delta : `~astropy.time.TimeDelta`
        Time in seconds after the MET reference.
    meta : dict
        Dictionary with the keywords ``MJDREFI`` and ``MJDREFF``.

    Returns
    -------
    time : `~astropy.time.Time`
        Absolute time with ``format='ISOT'`` and ``scale='UTC'``.
    """
    time = time_ref_from_dict(meta) + time_delta
    return Time(time.utc.isot)


def extract_time_info(row):
    """Extract the timing metadata from an event file header.

    Parameters
    ----------
    row : dict
        Dictionary with all the metadata of an event file.

    Returns
    -------
    time_row : dict
        Dictionary with only the time metadata.
    """
    time_row = {}
    for name in TIME_KEYWORDS:
        time_row[name] = row[name]
    return time_row


def unique_time_info(rows):
    """Check if the time information are identical between all metadata dictionaries.

    Parameters
    ----------
    rows : list
        List of dictionaries with a list of time metadata from different observations.

    Returns
    -------
    status : bool
        True if the time metadata are identical between observations.
    """
    if len(rows) <= 1:
        return True

    first_obs = rows[0]
    for row in rows[1:]:
        for name in TIME_KEYWORDS:
            if first_obs[name] != row[name] or row[name] is None:
                return False
    return True

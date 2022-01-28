# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time related utility functions."""
import numpy as np
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


def time_ref_to_dict(time, scale="tt"):
    """TODO: document and test.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Time object with ``format='MJD'``

    Returns
    -------
    meta : dict
        FITS time standard header info
    """
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

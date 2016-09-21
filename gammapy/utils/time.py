# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time related utility functions."""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.time import Time, TimeDelta

__all__ = [
    'time_ref_from_dict',
    'time_relative_to_ref',
    'absolute_time',
]

# TODO: implement and document this properly.
# see https://github.com/gammapy/gammapy/issues/284
TIME_REF_FERMI = Time('2001-01-01T00:00:00')


def time_ref_from_dict(meta):
    """Calculate the time reference from metadata.

    The time reference is built as MJDREFI + MJDREFF in units of MJD.
    All other times should be interpreted as seconds after the reference.

    Parameters
    ----------
    meta : `dict`
        dictionary with the keywords ``MJDREFI`` and ``MJDREFF``

    Returns
    -------
    time : `~astropy.time.Time`
        reference time with ``format='MJD'``
    """
    # Note: the `float` call here is to make sure we use 64-bit
    mjd = float(meta['MJDREFI']) + float(meta['MJDREFF'])
    # TODO: Is 'tt' a default we should put here?
    scale = meta.get('TIMESYS', 'tt').lower()
    # Note: we could call .copy('iso') or .replicate('iso')
    # here if we prefer 'iso' over 'mjd' format in most places.

    return Time(mjd, format='mjd', scale=scale)


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
    delta_time = TimeDelta(time - time_ref, format='sec')

    return delta_time


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
    time = Time(time.utc.isot)

    return time

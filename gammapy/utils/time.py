# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time related utility functions."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.time import Time, TimeDelta

__all__ = ['time_ref_from_dict',
           'time_relative_to_ref',
           ]


def time_ref_from_dict(meta):
    """Calculate the time reference from metadata.

    The time reference is built as MJDREFI + MJDREFF in units of MJD.
    All other times should be interpreted as seconds after the reference.

    Parameters
    ----------
    meta : `~dict`
    	dictionary with the keywords `MJDREFI` and `MJDREFF`

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
    time: `~astropy.time.Time`
    	time to be converted.
    meta : dict
    	dictionary with the keywords `MJDREFI` and `MJDREFF`

    Returns
    -------
    `~astropy.time.TimeDelta` object with the time in seconds after the reference.
    """
    time_ref = time_ref_from_dict(meta)
    delta_time = TimeDelta(time - time_ref, format='sec')

    return delta_time

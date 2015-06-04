# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time related utility functions."""
from astropy.time import Time

__all__ = ['time_ref_from_dict',
           ]


def time_ref_from_dict(meta):
    """ Calculte the time reference from metadata.
    The time reference is built as MJDREFI + MJDREFF in units of MJD.
    All other times should be interpreted as seconds after the reference.

    Parameters
    ----------
    meta : dict
    	dictionnary with the keywords 'MJDREFI' and 'MJDREFF'

    Returns
    -------
    `~astropy.Time` object with the reference time in units of MJD.
    """
    # Note: the `float` call here is to make sure we use 64-bit
    mjd = float(meta['MJDREFI']) + float(meta['MJDREFF'])
    # TODO: Is 'tt' a default we should put here?
    scale = meta.get('TIMESYS', 'tt').lower()
    # Note: we could call .copy('iso') or .replicate('iso')
    # here if we prefer 'iso' over 'mjd' format in most places.

    return Time(mjd, format='mjd', scale=scale)

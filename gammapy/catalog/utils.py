# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog utility functions / classes."""
from __future__ import print_function, division

__all__ = ['make_source_designation', 'parse_source_designation']


def make_source_designation(coordinate, ra_digits, acronym=''):
    """Compute source designation following the IAU specification.

    Following the IAU specification, the following formats are
    used for a given number of right ascension digits (`ra_digits`):
    
    --------- ------
    ra_digits format
    --------- ------
    4         HHMM+DDd
    5         HHMM.m+DDMM
    6         HHMMSS+DDMM.m
    7         HHMMSS.s+DDMMSS
    8         HHMMSS.ss+DDMMSS.s
    --------- ------------------
    
    Note that the declination part always has one digit less than
    the right ascension part. 
    
    Reference: http://cdsweb.u-strasbg.fr/Dic/iau-spec.html
    
    Parameters
    ----------
    coordinate : `astropy.coordinate.ICRS`
        Source position
    ra_digits : int (>= 4)
        Number of digits for the right ascension part
    acronym : str
        Source acronym (default: no acronym)
    
    Returns
    -------
    designation : str
        Source designation
    
    Examples
    --------
    >>> from astropy.coordinates import ICRS
    >>> from gammapy.catalog import make_source_designation
    
    Example: Crab pulsar position (positive declination)
    >>> coordinate = ICRS('05h34m31.93830s +22d00m52.1758s')
    >>> designation = make_source_designation(coordinate, ra_digits=4, acronym='HESS')
    >>> print(designation)
    HESS J0534+220
    
    Example: PKS 2155-304 AGN position (negative declination)
    >>> coordinate = ICRS('21h58m52.06511s -30d13m32.1182s')
    >>> designation = make_source_designation(coordinate, ra_digits=5, acronym='')
    >>> print(designation)
    J21588.7-3013
    """
    ra = coordinate.ra
    dec = coordinate.dec

    if ra_digits == 4: # format: HHMM+DDd
        ra_str = '{0:02d}{1:02d}'.format(int(ra.hms.h), int(ra.hms.m))
        dec_str = '{0:+03d}'.format(int(10 * dec.deg))
    elif ra_digits == 5: # format : HHMM.m+DDMM
        ra_str = '{0:02d}{1:02.1f}'.format(int(ra.hms.h), 10 * (ra.hms.m + ra.hms.s / 60.))
        dec_str = '{0:+02d}{1:02d}'.format(int(dec.dms.d), abs(int(dec.dms.m)))
    elif ra_digits >= 6:
        raise NotImplementedError()
    else:
        raise ValueError('Invalid ra_digits: {0}'.format(ra_digits))

    if acronym:
        designation = acronym + ' J' + ra_str + dec_str
    else:
        designation = 'J' + ra_str + dec_str

    return designation


def parse_source_designation(designation):
    raise NotImplementedError

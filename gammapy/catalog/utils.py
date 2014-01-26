# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog utility functions / classes."""
from __future__ import print_function, division

__all__ = ['coordinate_iau_format',
           'ra_iau_format',
           'dec_iau_format']


def coordinate_iau_format(coordinate, ra_digits, dec_digits=None):
    """Coordinate format as an IAU source designation.

    Reference: http://cdsweb.u-strasbg.fr/Dic/iau-spec.html
    
    Parameters
    ----------
    coordinate : `astropy.coordinate.ICRS`
        Source coordinate
    ra_digits : int (>=2)
        Number of digits for the Right Ascension part
    dec_digits : int (>=2) or None
        Number of digits for the declination part
        Default is `dec_digits` = None, meaning `dec_digits` = `ra_digits` - 1
    
    Returns
    -------
    strrepr : str
        IAU format string representation of the coordinate
    
    Examples
    --------
    >>> from astropy.coordinates import ICRS, FK4
    >>> from gammapy.catalog import coordinate_iau_format
    
    Example position from IAU specification

    >>> coordinate = ICRS('00h51m09.38s -42d26m33.8s')
    >>> designation = 'QSO J' + coordinate_iau_format(coordinate, ra_digits=6)
    >>> print(designation)
    QSO J005109-4226.5
    >>> designation = 'QSO B' + coordinate_iau_format(FK4(coordinate), ra_digits=6)
    >>> print(designation)
    QSO B004848-4242.8

    Example: Crab pulsar position (positive declination)

    >>> coordinate = ICRS('05h34m31.93830s +22d00m52.1758s')
    >>> designation = 'HESS J' + coordinate_iau_format(coordinate, ra_digits=4)
    >>> print(designation)
    HESS J0534+220
    
    Example: PKS 2155-304 AGN position (negative declination)

    >>> coordinate = ICRS('21h58m52.06511s -30d13m32.1182s')
    >>> designation = '2FGL J' + coordinate_iau_format(coordinate, ra_digits=5)
    >>> print(designation)
    2FGL J2158.8-3013
    """
    if dec_digits == None:
        dec_digits = max(2, ra_digits - 1) 
    
    ra_str = ra_iau_format(coordinate.ra, ra_digits)
    dec_str = dec_iau_format(coordinate.dec, dec_digits)

    return ra_str + dec_str


def ra_iau_format(ra, digits):
    """Right Ascension part of an IAU source designation.
    
    Reference: http://cdsweb.u-strasbg.fr/Dic/iau-spec.html

    ====== ========
    digits format
    ====== ========
    2      HH
    3      HHh
    4      HHMM
    5      HHMM.m
    6      HHMMSS
    7      HHMMSS.s
    ====== ========

    Parameters
    ----------
    ra : `astropy.coordinate.Longitude`
        Right ascension
    digits : int (>=2)
        Number of digits
    
    Returns
    -------
    strrepr : str
        IAU format string representation of the angle
    """
    if (not isinstance(digits, int)) and (digits >= 2): 
        raise ValueError('Invalid digits: {0}. Valid options: int >= 2'.format(digits))
    
    # Note that Python string formatting always rounds the last digit,
    # but the IAU spec requires to truncate instead.
    # That's why integers with the correct digits are computed and formatted
    # instead of formatting floats directly

    ra_h = int(ra.hms[0])
    ra_m = int(ra.hms[1])
    ra_s = ra.hms[2]

    if digits == 2:  # format: HH
        ra_str = '{0:02d}'.format(ra_h)
    elif digits == 3:  # format: HHh
        ra_str = '{0:03d}'.format(int(10 * ra.hour))
    elif digits == 4:  # format: HHMM
        ra_str = '{0:02d}{1:02d}'.format(ra_h, ra_m)
    elif digits == 5:  # format : HHMM.m
        ra_str = '{0:02d}{1:02d}.{2:01d}'.format(ra_h, ra_m, int(ra_s / 6))
    elif digits == 6:  # format: HHMMSS
        ra_str = '{0:02d}{1:02d}{2:02d}'.format(ra_h, ra_m, int(ra_s))
    else:  # format: HHMMSS.s
        SS = int(ra_s)
        s_digits = digits - 6
        s = int(10 ** s_digits * (ra_s - SS))
        fmt = '{0:02d}{1:02d}{2:02d}.{3:0' + str(s_digits) + 'd}'
        ra_str = fmt.format(ra_h, ra_m, SS, s)

    return ra_str


def dec_iau_format(dec, digits):
    """Declination part of an IAU source designation.
    
    Reference: http://cdsweb.u-strasbg.fr/Dic/iau-spec.html

    ====== =========
    digits format
    ====== =========
    2      +DD
    3      +DDd
    4      +DDMM
    5      +DDMM.m
    6      +DDMMSS
    7      +DDMMSS.s
    ====== =========

    Parameters
    ----------
    dec : `astropy.coordinate.Latitude`
        Declination
    digits : int (>=2)
        Number of digits
    
    Returns
    -------
    strrepr : str
        IAU format string representation of the angle
    """
    if not isinstance(digits, int) and digits >= 2: 
        raise ValueError('Invalid digits: {0}. Valid options: int >= 2'.format(digits))
    
    # Note that Python string formatting always rounds the last digit,
    # but the IAU spec requires to truncate instead.
    # That's why integers with the correct digits are computed and formatted
    # instead of formatting floats directly

    dec_sign = '+' if dec.deg >= 0 else '-'
    dec_d = int(abs(dec.dms[0]))
    dec_m = int(abs(dec.dms[1]))
    dec_s = abs(dec.dms[2])

    if digits == 2:  # format: +DD
        dec_str = '{0}{1:02d}'.format(dec_sign, dec_d)
    elif digits == 3:  # format: +DDd
        dec_str = '{0:+04d}'.format(int(10 * dec.deg))
    elif digits == 4:  # format : +DDMM
        dec_str = '{0}{1:02d}{2:02d}'.format(dec_sign, dec_d, dec_m)
    elif digits == 5:  # format: +DDMM.m
        dec_str = '{0}{1:02d}{2:02d}.{3:01d}'.format(dec_sign, dec_d, dec_m, int(dec_s / 6))
    elif digits == 6:  # format: +DDMMSS
        dec_str = '{0}{1:02d}{2:02d}.{3:02d}'.format(dec_sign, dec_d, dec_m, int(dec_s))
    else:  # format: +DDMMSS.s
        SS = int(dec_s)
        s_digits = digits - 6
        s = int(10 ** s_digits * (dec_s - SS))
        fmt = '{0}{1:02d}{2:02d}{3:02d}.{4:0' + str(s_digits) + 'd}'
        dec_str = fmt.format(dec_sign, dec_d, dec_m, SS, s)

    return dec_str

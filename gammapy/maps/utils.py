# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.io import fits


def unpack_seq(seq, n=1):
    """Utility to unpack the first N values of a tuple or list.  Remaining
    values are put into a single list which is the last element of the
    return value.  This partially simulates the extended unpacking
    functionality available in Python 3.

    Parameters
    ----------
    seq : list or tuple
        Input sequence to be unpacked.

    n : int
        Number of elements of ``seq`` to unpack.  Remaining elements
        are put into a single tuple.

    """

    for row in seq:
        yield [e for e in row[:n]] + [row[n:]]


def find_bands_hdu(hdulist, hdu):
    """Discover the extension name of the BANDS HDU.

    Returns
    -------
    extname : str
        Extension name of the BANDS HDU.  None if no BANDS HDU was found.
    """
    if 'BANDSHDU' in hdu.header:
        return hdu.header['BANDSHDU']

    has_cube_data = False

    if (isinstance(hdu, (fits.ImageHDU, fits.PrimaryHDU)) and
            hdu.header.get('NAXIS', None) == 3):
        has_cube_data = True
    elif isinstance(hdu, fits.BinTableHDU):

        if (hdu.header.get('INDXSCHM', '') == 'IMPLICIT' and
                len(hdu.columns) > 1):
            has_cube_data = True

    if has_cube_data:
        if 'EBOUNDS' in hdulist:
            return 'EBOUNDS'
        elif 'ENERGIES' in hdulist:
            return 'ENERGIES'

    return None


def find_hdu(hdulist):
    """Find the first non-empty HDU."""

    for hdu in hdulist:
        if hdu.data is not None:
            return hdu

    raise AttributeError('No Image or BinTable HDU found.')


def find_image_hdu(hdulist):

    for hdu in hdulist:
        if hdu.data is not None and isinstance(hdu, fits.ImageHDU):
            return hdu

    raise AttributeError('No Image HDU found.')


def find_bintable_hdu(hdulist):

    for hdu in hdulist:
        if hdu.data is not None and isinstance(hdu, fits.BinTableHDU):
            return hdu

    raise AttributeError('No BinTable HDU found.')

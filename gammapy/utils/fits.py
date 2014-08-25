# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FITS utility functions.
"""
from __future__ import print_function, division
from astropy.io import fits

__all__ = ['get_hdu',
           'get_image_hdu',
           'get_table_hdu',
           'fits_table_to_pandas',
           ]


def get_hdu(location):
    """Get one HDU for a given location.

    location should be either a ``file_name`` or a file
    and HDU name in the format ``file_name[hdu_name]``.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    # TODO: Test all cases and give good exceptions / error messages
    if '[' in location:
        tokens = location.split('[')
        file_name = tokens[0]
        hdu_name = tokens[1][:-1]  # split off ']' at the end
        return fits.open(file_name)[hdu_name]
    else:
        file_name = location
        return fits.open(file_name)[0]


def get_image_hdu():
    """Get the first image HDU."""
    raise NotImplementedError


def get_table_hdu():
    """Get the first table HDU."""
    raise NotImplementedError


def fits_table_to_pandas(filename, index_columns):
    """Convert a FITS table HDU to a `pandas.DataFrame`.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    # TODO: really make this work for an astropy Table (not a TableHDU or filename).
    from pandas import DataFrame
    data = fits.getdata(filename)

    # Convert byte order to native ... this is required by pandas
    # See https://github.com/astropy/astropy/issues/1156
    # and http://pandas.pydata.org/pandas-docs/dev/faq.html#byte-ordering-issues
    data = data.byteswap().newbyteorder()
    table = DataFrame(data)

    # Strip whitespace for string columns that will become indices
    for index_column in index_columns:
        table[index_column].map(str.strip)
    table = table.set_index(index_columns)

    return table

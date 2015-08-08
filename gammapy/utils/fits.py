# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FITS utility functions.
"""
from __future__ import print_function, division
from astropy.io import fits

__all__ = ['get_hdu',
           'get_image_hdu',
           'get_table_hdu',
           'fits_table_to_pandas',
           'table_to_fits_table',
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


def table_to_fits_table(table):
    """Convert astropy table to binary table fits format.

    This is a generic method to convert a `~astropy.table.Table`
    to a `~astropy.io.fits.BinTableHDU`.
    The name of the table can be stored in the Table meta information
    under the `name` keyword.

    Parameters
    ----------
    table : `~astropy.table.Table`
        astropy table containing the desired columns

    Returns
    -------
    tbhdu : `~astropy.io.fits.BinTableHDU`
        fits bin table containing the astropy table columns
    """
    # read name and drop it from the meta information, otherwise
    # it would be stored as a header keyword in the BinTableHDU
    if 'name' in table.meta:
        name = table.meta.popitem('name')[1]
    else:
        name = None

    data = table.as_array()

    header = fits.Header()
    header.update(table.meta)

    tbhdu = fits.BinTableHDU(data, header, name=name)

    # Copy over column meta-data
    for colname in table.colnames:
        tbhdu.columns[colname].unit = str(table[colname].unit)

    # TODO: this method works fine but the order of keywords in the table
    # header is not logical: for instance, list of keywords with column
    # units (TUNITi) is appended after the list of column keywords
    # (TTYPEi, TFORMi), instead of in between.
    # As a matter of fact, the units aren't yet in the header, but
    # only when calling the write method and opening the output file.
    # https://github.com/gammapy/gammapy/issues/298

    return tbhdu


def fits_table_to_table(tbhdu):
    """Convert astropy table to binary table fits format.

    This is a generic method to convert a `~astropy.io.fits.BinTableHDU`
    to `~astropy.table.Table`.
    to a 
    The name of the table is stored in the Table meta information
    under the `name` keyword.

    Parameters
    ----------
    tbhdu : `~astropy.io.fits.BinTableHDU`
        fits bin table containing the astropy table columns

    Returns
    -------
    table : `~astropy.table.Table`
        astropy table containing the desired columns
    """

    data = tbhdu.data
    header = tbhdu.header

    table = Table(data, meta=header)

    # Copy over column meta-data
    for colname in tbhdu.columns.names:
        table[colname].unit = tbhdu.columns[colname].unit

    return table

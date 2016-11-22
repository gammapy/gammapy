# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FITS utility functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from astropy.io import fits
from astropy.table import Table, QTable

__all__ = [
    'table_from_row_data',
    'get_hdu',
    'table_to_fits_table',
    'fits_table_to_table',
    'energy_axis_to_ebounds',
]


def table_from_row_data(rows, type='qtable', **kwargs):
    """Helper function to create table objects from row data.

    - Works with quantities.
    - Preserves order of keys if OrderedDicts are used.

    Parameters
    ----------
    rows : list
        List of row data (each row a dict or OrderedDict)
    type : {'table', 'qtable'}
        Type of table to create
    """
    # Creating `QTable` from list of row data with `Quantity` objects
    # doesn't work. So we're reformatting to list of column `Quantity`
    # objects here.
    # table = QTable(rows=rows)

    if type == 'table':
        cls = Table
    elif type == 'qtable':
        cls = QTable
    else:
        raise ValueError('Invalid type: {}'.format(type))

    table = cls(**kwargs)
    colnames = list(rows[0].keys())
    for name in colnames:
        coldata = [_[name] for _ in rows]
        if isinstance(rows[0][name], Quantity):
            coldata = Quantity(coldata, unit=rows[0][name].unit)
        table[name] = coldata

    return table


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


def table_to_fits_table(table):
    """Convert `~astropy.table.Table` to `astropy.io.fits.BinTableHDU`.

    The name of the table can be stored in the Table meta information
    under the ``name`` keyword.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table

    Returns
    -------
    hdu : `~astropy.io.fits.BinTableHDU`
        Binary table HDU
    """
    # read name and drop it from the meta information, otherwise
    # it would be stored as a header keyword in the BinTableHDU
    name = table.meta.pop('name', None)

    table.convert_unicode_to_bytestring(python3_only=True)
    data = table.as_array()

    header = fits.Header()
    header.update(table.meta)

    hdu = fits.BinTableHDU(data, header, name=name)

    # Copy over column meta-data
    for colname in table.colnames:
        if table[colname].unit is not None:
            hdu.columns[colname].unit = table[colname].unit.to_string('fits')

    # TODO: this method works fine but the order of keywords in the table
    # header is not logical: for instance, list of keywords with column
    # units (TUNITi) is appended after the list of column keywords
    # (TTYPEi, TFORMi), instead of in between.
    # As a matter of fact, the units aren't yet in the header, but
    # only when calling the write method and opening the output file.
    # https://github.com/gammapy/gammapy/issues/298

    return hdu


def fits_table_to_table(tbhdu):
    """Convert astropy table to binary table FITS format.

    This is a generic method to convert a `~astropy.io.fits.BinTableHDU`
    to `~astropy.table.Table`.
    The name of the table is stored in the Table meta information
    under the ``name`` keyword.

    Parameters
    ----------
    hdu : `~astropy.io.fits.BinTableHDU`
        FITS bin table containing the astropy table columns

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


def energy_axis_to_ebounds(energy):
    """Convert energy `~gammapy.utils.nddata.BinnedEnergyAxis` to OGIP
    ``EBOUNDS`` extension

    see
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.2
    """
    table = Table()

    table['CHANNEL'] = np.arange(energy.nbins, dtype=np.int16)
    table['E_MIN'] = energy.data[:-1]
    table['E_MAX'] = energy.data[1:]

    hdu = table_to_fits_table(table)

    header = hdu.header
    header['EXTNAME'] = 'EBOUNDS', 'Name of this binary table extension'
    header['TELESCOP'] = 'DUMMY', 'Mission/satellite name'
    header['INSTRUME'] = 'DUMMY', 'Instrument/detector'
    header['FILTER'] = 'None', 'Filter information'
    header['CHANTYPE'] = 'PHA', 'Type of channels (PHA, PI etc)'
    header['DETCHANS'] = energy.nbins, 'Total number of detector PHA channels'
    header['HDUCLASS'] = 'OGIP', 'Organisation devising file format'
    header['HDUCLAS1'] = 'RESPONSE', 'File relates to response of instrument'
    header['HDUCLAS2'] = 'EBOUNDS', 'This is an EBOUNDS extension'
    header['HDUVERS'] = '1.2.0', 'Version of file format'

    return hdu


def ebounds_to_energy_axis(ebounds):
    """Convert ``EBOUNDS`` extension to energy
    `~gammapy.utils.nddata.BinnedEnergyAxis`
    """
    from .nddata import BinnedDataAxis
    table = fits_table_to_table(ebounds)
    emin = table['E_MIN'].quantity
    emax = table['E_MAX'].quantity
    energy = np.append(emin.value, emax.value[-1]) * emin.unit
    return BinnedDataAxis(data=energy)

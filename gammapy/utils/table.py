# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Table helper utilities.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from astropy.table import Table
from astropy.units import Quantity
from .units import standardise_unit

__all__ = [
    'table_standardise_units_copy',
    'table_standardise_units_inplace',
    'table_row_to_dict',
    'table_from_row_data',
]


def table_standardise_units_copy(table):
    """Standardise units for all columns in a table in a copy.

    Calls `~gammapy.utils.units.standardise_unit`.

    Parameters
    ----------
    table : `~astropy.table.Table` or `~astropy.table.QTable`
        Input table (won't be modified)

    Returns
    -------
    table : `~astropy.table.Table`
        Copy of the input table with standardised column units
    """
    # Note: we could add an `inplace` option (or variant of this function)
    # for `Table`, but not for `QTable`.
    # See https://github.com/astropy/astropy/issues/6098
    table = Table(table)
    return table_standardise_units_inplace(table)


def table_standardise_units_inplace(table):
    """Standardise units for all columns in a table in place.
    """
    for column in table.columns.values():
        if column.unit:
            column.unit = standardise_unit(column.unit)

    return table


def table_row_to_dict(row, make_quantity=True):
    """Make one source data dict.

    Parameters
    ----------
    row : `~astropy.table.Row`
        Row
    make_quantity : bool
        Make quantity values for columns with units

    Returns
    -------
    data : `~collections.OrderedDict`
        Row data
    """
    data = OrderedDict()
    for name, col in row.columns.items():
        val = row[name]
        if make_quantity and col.unit:
            val = Quantity(val, unit=col.unit)
        data[name] = val
    return data


# TODO: remove type = 'qtable' to avoid issues?
# see https://github.com/astropy/astropy/issues/6098
# see https://github.com/gammapy/gammapy/issues/980
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



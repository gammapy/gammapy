# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Table helper utilities."""
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from .units import standardise_unit

__all__ = [
    "hstack_columns",
    "table_row_to_dict",
    "table_standardise_units_copy",
    "table_standardise_units_inplace",
]


def hstack_columns(table, table_other):
    """Stack the column data horizontally.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Input table.
    table_other : `~astropy.table.Table`
        Other input table.

    Returns
    -------
    stacked : `~astropy.table.Table`
        Stacked table.
    """
    stacked = Table()

    for column in table.colnames:
        data = np.hstack([table[column].data[0], table_other[column].data[0]])
        stacked[column] = data[np.newaxis, :]
    return stacked


def table_standardise_units_copy(table):
    """Standardise units for all columns in a table in a copy.

    Calls `~gammapy.utils.units.standardise_unit`.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Input table (won't be modified).

    Returns
    -------
    table : `~astropy.table.Table`
        Copy of the input table with standardised column units.
    """
    # Note: we could add an `inplace` option (or variant of this function)
    # See https://github.com/astropy/astropy/issues/6098
    table = Table(table)
    return table_standardise_units_inplace(table)


def table_standardise_units_inplace(table):
    """Standardise units for all columns in a table in place."""
    for column in table.columns.values():
        if column.unit:
            column.unit = standardise_unit(column.unit)

    return table


def table_row_to_dict(row, make_quantity=True):
    """Make one source data dictionary.

    Parameters
    ----------
    row : `~astropy.table.Row`
        Row.
    make_quantity : bool, optional
        Make quantity values for columns with units.
        Default is True.

    Returns
    -------
    data : dict
        Row data.
    """
    data = {}
    for name, col in row.columns.items():
        val = row[name]

        if make_quantity and col.unit:
            val = Quantity(val, unit=col.unit)
        data[name] = val
    return data

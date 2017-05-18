# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Table helper utilities.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table
from .units import standardise_unit

__all__ = [
    'table_standardise_units_copy',
    'table_standardise_units_inplace',
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

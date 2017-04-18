# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Table helper utilities.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from .units import standardise_unit

__all__ = [
    'table_standardise',
]


def table_standardise(table):
    """TODO
    """
    for column in table.columns.values():
        if column.unit:
            column.unit = standardise_unit(column.unit)

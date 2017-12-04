# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from collections import OrderedDict
import astropy.units as u
from astropy.table import Table, QTable, Column
from ..table import table_standardise_units_copy, table_row_to_dict, table_from_row_data


@pytest.mark.parametrize('table_class', [Table, QTable])
def test_table_standardise_units(table_class):
    table = table_class()
    table['a'] = Column([1], unit='ph cm-2 s-1')
    table['b'] = Column([1], unit='ct cm-2 s-1')
    table['c'] = Column([1], unit='cm-2 s-1')
    table['d'] = Column([1])

    table = table_standardise_units_copy(table)

    assert table['a'].unit == u.Unit('cm-2 s-1')
    assert table['b'].unit == u.Unit('cm-2 s-1')
    assert table['c'].unit == u.Unit('cm-2 s-1')
    assert table['d'].unit is None


@pytest.fixture()
def table():
    return Table([
        Column([1, 2], 'col_a'),
        Column([1, 2] * u.m, 'col_c'),
        Column(['x', 'yy'], 'col_d'),
    ])


def test_table_row_to_dict(table):
    actual = table_row_to_dict(table[1])
    expected = OrderedDict([
        ('col_a', 2),
        ('col_c', 2 * u.m),
        ('col_d', 'yy'),
    ])
    assert actual == expected

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
from astropy.table import Table, Column
from ..table import table_standardise


def test_table_standardise():
    table = Table()
    table['a'] = Column([1], unit='ph cm-2 s-1')
    table['b'] = Column([1], unit='ct cm-2 s-1')
    table['c'] = Column([1], unit='cm-2 s-1')
    table['d'] = Column([1])

    table_standardise(table)

    assert table['a'].unit == u.Unit('cm-2 s-1')
    assert table['b'].unit == u.Unit('cm-2 s-1')
    assert table['c'].unit == u.Unit('cm-2 s-1')
    assert table['d'].unit is None

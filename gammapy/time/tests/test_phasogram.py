# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
from ...utils.testing import requires_dependency
from ..phasogram import Phasogram


def make_test_phasogram():
    table = Table()
    table['PHASE_MIN'] = [0, 0.2, 0.7]
    table['PHASE_MAX'] = [0.2, 0.7, 1]
    table['COUNTS'] = [7, 0, 2]
    return Phasogram(table)


def test_phasogram():
    phasogram = make_test_phasogram()
    assert len(phasogram.table) == 3

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
from ...utils.testing import requires_dependency
from ..phasogram import Phasogram
from ...data.event_list import EventList

def make_test_eventlist():
    table = Table()
    table['PHASE'] = [0.1, 0.25, 0.71, 0.85]
    return EventList(table)

def make_test_phasogram():
    table = Table()
    table['PHASE_MIN'] = [0, 0.2, 0.7]
    table['PHASE_MAX'] = [0.2, 0.7, 1]
    table['VALUE'] = [7, 0, 2]
    return Phasogram(table)


def test_phasogram_init():
    phasogram = make_test_phasogram()
    assert len(phasogram.table) == 3
    assert_allclose(phasogram.phase_bins, [0, 0.2, 0.7, 1])

def test_phasogram_fill_events():
    phasogram = make_test_phasogram()
    events = make_test_eventlist()
    phasogram.fill_events(events)
    assert_allclose(phasogram.table['VALUE'], [1, 1, 2])

test_phasogram_fill_events()
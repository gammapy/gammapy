# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ...data import EventList, EventListDataset, EventListDatasetChecker
from ...datasets import get_path


filename = get_path('hess/run_0023037_hard_eventlist.fits.gz')


def test_EventList():
    event_list = EventList.read(filename, hdu='EVENTS')

    assert len(event_list) == 49
    assert isinstance(event_list.info, str)


def test_EventListDataset():
    dset = EventListDataset.read(filename)

    assert len(dset.event_list) == 49
    assert isinstance(dset.info, str)


def test_EventListDatasetChecker():
    dset = EventListDataset.read(filename)
    checker = EventListDatasetChecker(dset)
    # checker.check('all')

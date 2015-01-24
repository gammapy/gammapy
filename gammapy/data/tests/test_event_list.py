# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from ...data import EventListDataset
from ...datasets import get_path


def test_EventList():
    filename = get_path('hess/run_0023037_hard_eventlist.fits.gz')
    ds = EventListDataset.read(filename)

    assert len(ds.event_list) == 49

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ...data import EventList, EventListDataset, EventListDatasetChecker
from ...datasets import get_path


filename = get_path('hess/run_0023037_hard_eventlist.fits.gz')


def test_EventList():
    event_list = EventList.read(filename, hdu='EVENTS')

    assert len(event_list) == 49
    assert 'Event list info' in event_list.info
    assert event_list.time[0].iso == '2004-10-14 00:08:32.000'
    assert event_list.radec[0].to_string() == '82.7068 19.8186'
    assert event_list.altaz[0].to_string() == '46.2059 31.2001'
    assert '{:1.5f}'.format(event_list.energy[0]) == '11.64355 TeV'

    lon, lat, height = event_list.observatory_earth_location.to_geodetic()
    assert '{:1.5f}'.format(lon) == '16.50022 deg'
    assert '{:1.5f}'.format(lat) == '-23.27178 deg'
    assert '{:1.5f}'.format(height) == '1835.00000 m'


def test_EventListDataset():
    dset = EventListDataset.read(filename)

    assert len(dset.event_list) == 49
    assert 'Event list dataset info' in dset.info


def test_EventListDatasetChecker():
    dset = EventListDataset.read(filename)
    checker = EventListDatasetChecker(dset)
    checker.run('all')

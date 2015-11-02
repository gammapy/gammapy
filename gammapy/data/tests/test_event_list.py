# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...utils.testing import requires_data
from ...data import EventList, EventListDataset, EventListDatasetChecker
from ...datasets import gammapy_extra
import numpy as np

@requires_data('gammapy-extra')
def test_EventList():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    event_list = EventList.read(filename, hdu='EVENTS')

    assert len(event_list) == 49
    assert 'Event list info' in event_list.summary
    assert event_list.time[0].iso == '2004-10-14 00:08:39.214'
    assert event_list.radec[0].to_string() == '82.7068 19.8186'
    assert event_list.galactic[0].to_string() == '185.956 -7.69277'
    assert event_list.altaz[0].to_string() == '46.2059 31.2001'
    assert '{:1.5f}'.format(event_list.energy[0]) == '11.64355 TeV'

    lon, lat, height = event_list.observatory_earth_location.to_geodetic()
    assert '{:1.5f}'.format(lon) == '16.50022 deg'
    assert '{:1.5f}'.format(lat) == '-23.27178 deg'
    assert '{:1.5f}'.format(height) == '1835.00000 m'

    assert '{:1.5f}'.format(event_list.observation_time_duration) == '1577.00000 s'
    assert '{:1.5f}'.format(event_list.observation_live_time_duration) == '1510.95911 s'
    assert_allclose(event_list.observation_dead_time_fraction, 0.03576320037245795)


@requires_data('gammapy-extra')
def test_EventList_region():
    from gammapy.background import CircularOffRegions

    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    event_list = EventList.read(filename, hdu='EVENTS')

    regions_file = gammapy_extra.filename('test_datasets/background/example_off_regions.dat')
    off = CircularOffRegions.read(regions_file, format='ascii.ecsv')
    filter = event_list.filter_circular_regions(off)
    assert len(filter) == 5

@requires_data('gammapy-extra')
def test_EventListDataset():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    dset = EventListDataset.read(filename)

    assert len(dset.event_list) == 49
    assert 'Event list dataset info' in dset.info
    # TODO: test all methods ... get ~ 100% test coverage
    # even without running the following test.


@requires_data('gammapy-extra')
def test_EventListDatasetChecker():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    dset = EventListDataset.read(filename)
    checker = EventListDatasetChecker(dset)
    checker.run('all')


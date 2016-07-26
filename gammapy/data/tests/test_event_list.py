# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from ...utils.testing import requires_dependency, requires_data
from ...data import EventList, EventListDataset, EventListDatasetChecker
from ...datasets import gammapy_extra


@requires_data('gammapy-extra')
def test_EventList():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    event_list = EventList.read(filename)
    event_list.summary()

    assert len(event_list) == 49
    assert event_list.time[0].iso == '2004-10-14 00:08:39.214'
    assert event_list.radec[0].to_string() == '82.7068 19.8186'
    assert event_list.galactic[0].to_string(precision=2) == '185.96 -7.69'
    assert event_list.altaz[0].to_string() == '46.2059 31.2001'
    assert_allclose(event_list.offset[0].value, 1.904497742652893, rtol=1e-5)
    assert '{:1.5f}'.format(event_list.energy[0]) == '11.64355 TeV'

    lon, lat, height = event_list.observatory_earth_location.to_geodetic()
    assert '{:1.5f}'.format(lon) == '16.50022 deg'
    assert '{:1.5f}'.format(lat) == '-23.27178 deg'
    assert '{:1.5f}'.format(height) == '1835.00000 m'


@requires_data('gammapy-extra')
def test_EventList_region():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    event_list = EventList.read(filename, hdu='EVENTS')

    pos = SkyCoord(81, 21, unit='deg', frame='icrs')
    radius = Angle(1, 'deg')
    circ = CircleSkyRegion(pos, radius)
    idx = circ.contains(event_list.radec)
    filtered_list = event_list[idx]

    assert_allclose(filtered_list[4]['RA'], 81, rtol=1)
    assert_allclose(filtered_list[2]['DEC'], 21, rtol=1)
    assert len(filtered_list) == 5


@requires_data('gammapy-extra')
def test_EventListDataset():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    dset = EventListDataset.read(filename)
    dset.info()

    assert len(dset.event_list) == 49
    # TODO: test all methods ... get ~ 100% test coverage
    # even without running the following test.


@requires_data('gammapy-extra')
def test_EventListDatasetChecker():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    dset = EventListDataset.read(filename)
    checker = EventListDatasetChecker(dset)
    checker.run('all')


@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_Eventlist_peek():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    event_list = EventList.read(filename, hdu='EVENTS')

    event_list.peek()

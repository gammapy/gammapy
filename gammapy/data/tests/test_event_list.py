# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from ...utils.testing import requires_dependency, requires_data
from ...data import EventList, EventListLAT, EventListDataset, EventListDatasetChecker


@requires_data('gammapy-extra')
class TestEventListHESS:
    def setup(self):
        filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz'
        self.events = EventList.read(filename)

    def test_basics(self):
        assert 'EventList' in str(self.events)

        assert len(self.events.table) == 49
        assert self.events.time[0].iso == '2004-10-14 00:08:39.214'
        assert self.events.radec[0].to_string() == '82.7068 19.8186'
        assert self.events.galactic[0].to_string(precision=2) == '185.96 -7.69'
        assert self.events.altaz[0].to_string() == '46.2059 31.2001'
        assert_allclose(self.events.offset[0].value, 1.904497742652893, rtol=1e-5)
        assert '{:1.5f}'.format(self.events.energy[0]) == '11.64355 TeV'

        lon, lat, height = self.events.observatory_earth_location.to_geodetic()
        assert '{:1.5f}'.format(lon) == '16.50022 deg'
        assert '{:1.5f}'.format(lat) == '-23.27178 deg'
        assert '{:1.5f}'.format(height) == '1835.00000 m'

    def test_stack(self):
        event_lists = [self.events] * 3
        stacked_list = EventList.stack(event_lists)
        assert len(stacked_list.table) == 49 * 3

    @requires_dependency('matplotlib')
    def test_peek(self):
        self.events.peek()

    @requires_dependency('matplotlib')
    def test_plot_offset2_distribution(self):
        self.events.plot_offset2_distribution()


@requires_data('gammapy-extra')
class TestEventListFermi:
    def setup(self):
        filename = '$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz'
        self.events = EventListLAT.read(filename)

    def test_basics(self):
        assert 'EventList' in str(self.events)

    @requires_dependency('matplotlib')
    def test_plot_image(self):
        self.events.plot_image()


@requires_data('gammapy-extra')
class TestEventListDataset:
    def test(self):
        filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz'
        dset = EventListDataset.read(filename)
        assert 'Event list dataset info' in str(dset)

        assert len(dset.event_list.table) == 49
        # TODO: test all methods ... get ~ 100% test coverage
        # even without running the following test.


@pytest.mark.xfail
@requires_data('gammapy-extra')
class TestEventListDatasetChecker():
    def test(self):
        filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz'
        dset = EventListDataset.read(filename)
        checker = EventListDatasetChecker(dset)
        checker.run('all')

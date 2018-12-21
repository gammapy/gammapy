# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from ...data import DataStore
from ...utils.testing import requires_data
from ...utils.testing import assert_time_allclose, assert_skycoord_allclose


@pytest.fixture(scope="session")
def data_store():
    return DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/")


@requires_data("gammapy-extra")
def test_data_store_observation(data_store):
    """Test DataStoreObservation class"""
    obs = data_store.obs(23523)

    assert_time_allclose(obs.tstart, Time(53343.92234009259, scale="tt", format="mjd"))
    assert_time_allclose(obs.tstop, Time(53343.94186555556, scale="tt", format="mjd"))

    c = SkyCoord(83.63333129882812, 21.51444435119629, unit="deg")
    assert_skycoord_allclose(obs.pointing_radec, c)

    c = SkyCoord(22.481705, 41.38979, unit="deg")
    assert_skycoord_allclose(obs.pointing_altaz, c)

    c = SkyCoord(83.63333129882812, 22.01444435119629, unit="deg")
    assert_skycoord_allclose(obs.target_radec, c)


@requires_data("gammapy-extra")
@pytest.mark.parametrize(
    "time_interval, expected_times",
    [
        (Time(["2004-12-04T22:10:00", "2004-12-04T22:30:00"], format="isot", scale="tt"),
         Time(["2004-12-04T22:10:00", "2004-12-04T22:30:00"], format="isot", scale="tt")),
        (Time([53343.930, 53343.940], format="mjd", scale="tt"),
         Time([53343.930, 53343.940], format="mjd", scale="tt")),
        (Time([10., 100000.], format='mjd', scale='tt'),
         Time([53343.92234009, 53343.94186563], format='mjd', scale='tt')),
        (Time([10., 20.], format='mjd', scale='tt'), None),
    ],
)
def test_observation_select_time(data_store, time_interval, expected_times):
    obs = data_store.obs(23523)
    print(obs.events.time[-1])
    print(obs.gti.time_stop[-1])

    new_obs = obs.select_time(time_interval)

    if expected_times:
        expected_times.format = 'mjd'
        assert np.all((new_obs.events.time >= expected_times[0]) & (new_obs.events.time < expected_times[1]))
        assert_time_allclose(new_obs.gti.time_start[0], expected_times[0], atol=0.01)
        assert_time_allclose(new_obs.gti.time_stop[-1], expected_times[1], atol=0.01)
    else:
        assert len(new_obs.events.table) == 0
        assert len(new_obs.gti.table) == 0


@requires_data("gammapy-extra")
class TestObservationChecker:
    def setup(self):
        data_store = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps")
        self.observation = data_store.obs(111140)

    def test_check_all(self):
        records = list(self.observation.check())
        assert len(records) == 8

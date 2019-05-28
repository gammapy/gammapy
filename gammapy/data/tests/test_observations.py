# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from ...data import DataStore
from ...utils.testing import requires_data
from ...utils.testing import assert_time_allclose, assert_skycoord_allclose


@pytest.fixture(scope="session")
def data_store():
    return DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")


@requires_data()
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


@requires_data()
@pytest.mark.parametrize(
    "time_interval, expected_times",
    [
        (
            Time(
                ["2004-12-04T22:10:00", "2004-12-04T22:30:00"],
                format="isot",
                scale="tt",
            ),
            Time(
                ["2004-12-04T22:10:00", "2004-12-04T22:30:00"],
                format="isot",
                scale="tt",
            ),
        ),
        (
            Time([53343.930, 53343.940], format="mjd", scale="tt"),
            Time([53343.930, 53343.940], format="mjd", scale="tt"),
        ),
        (
            Time([10.0, 100000.0], format="mjd", scale="tt"),
            Time([53343.92234009, 53343.94186563], format="mjd", scale="tt"),
        ),
        (Time([10.0, 20.0], format="mjd", scale="tt"), None),
    ],
)
def test_observation_select_time(data_store, time_interval, expected_times):
    obs = data_store.obs(23523)

    new_obs = obs.select_time(time_interval)

    if expected_times:
        expected_times.format = "mjd"
        assert np.all(
            (new_obs.events.time >= expected_times[0])
            & (new_obs.events.time < expected_times[1])
        )
        assert_time_allclose(new_obs.gti.time_start[0], expected_times[0], atol=0.01)
        assert_time_allclose(new_obs.gti.time_stop[-1], expected_times[1], atol=0.01)
    else:
        assert len(new_obs.events.table) == 0
        assert len(new_obs.gti.table) == 0


@requires_data()
@pytest.mark.parametrize(
    "time_interval, expected_times, expected_nr_of_obs",
    [
        (
            Time([53090.130, 53090.140], format="mjd", scale="tt"),
            Time([53090.130, 53090.140], format="mjd", scale="tt"),
            1,
        ),
        (
            Time([53090.130, 53091.110], format="mjd", scale="tt"),
            Time([53090.130, 53091.110], format="mjd", scale="tt"),
            3,
        ),
        (
            Time([10.0, 53111.0230], format="mjd", scale="tt"),
            Time([53090.1234512, 53111.0230], format="mjd", scale="tt"),
            8,
        ),
        (Time([10.0, 20.0], format="mjd", scale="tt"), None, 0),
    ],
)
def test_observations_select_time(
    data_store, time_interval, expected_times, expected_nr_of_obs
):
    obs_ids = data_store.obs_table["OBS_ID"][:8]
    obss = data_store.get_observations(obs_ids)

    new_obss = obss.select_time(time_interval)

    assert len(new_obss) == expected_nr_of_obs

    if expected_nr_of_obs > 0:
        assert new_obss[0].events.time[0] >= expected_times[0]
        assert new_obss[-1].events.time[-1] < expected_times[1]
        assert_time_allclose(
            new_obss[0].gti.time_start[0], expected_times[0], atol=0.01
        )
        assert_time_allclose(
            new_obss[-1].gti.time_stop[-1], expected_times[1], atol=0.01
        )


@requires_data()
class TestObservationChecker:
    def setup(self):
        data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")
        self.observation = data_store.obs(111140)

    def test_check_all(self):
        records = list(self.observation.check())
        assert len(records) == 8

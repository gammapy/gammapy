# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gammapy.data import DataStore, Observation
from gammapy.irf import load_cta_irfs
from gammapy.utils.testing import (
    assert_skycoord_allclose,
    assert_time_allclose,
    mpl_plot_check,
    requires_data,
    requires_dependency,
)


@pytest.fixture(scope="session")
def data_store():
    return DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")


@requires_data()
def test_observation(data_store):
    """Test Observation class"""
    obs = data_store.obs(23523)

    assert_time_allclose(obs.tstart, Time(53343.92234009259, scale="tt", format="mjd"))
    assert_time_allclose(obs.tstop, Time(53343.94186555556, scale="tt", format="mjd"))

    c = SkyCoord(83.63333129882812, 21.51444435119629, unit="deg")
    assert_skycoord_allclose(obs.pointing_radec, c)

    c = SkyCoord(22.481705, 41.38979, unit="deg")
    assert_skycoord_allclose(obs.pointing_altaz, c)

    c = SkyCoord(83.63333129882812, 22.01444435119629, unit="deg")
    assert_skycoord_allclose(obs.target_radec, c)


@requires_dependency("matplotlib")
@requires_data()
def test_observation_peek(data_store):
    obs = data_store.obs(23523)

    with mpl_plot_check():
        obs.peek()

    obs.bkg = None
    with mpl_plot_check():
        obs.peek()


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
def test_observations_mutation(data_store):
    obs_ids = data_store.obs_table["OBS_ID"][:4]
    obss = data_store.get_observations(obs_ids)
    assert obss.ids == ["20136", "20137", "20151", "20275"]

    obs_id = data_store.obs_table["OBS_ID"][4]
    obs = data_store.get_observations([obs_id])[0]

    obss.append(obs)
    assert obss.ids == ["20136", "20137", "20151", "20275", "20282"]
    obss.insert(0, obs)
    assert obss.ids == ["20282", "20136", "20137", "20151", "20275", "20282"]
    obss.pop(0)
    assert obss.ids == ["20136", "20137", "20151", "20275", "20282"]
    obs3 = obss[3]
    obss.pop(obss.ids[3])
    assert obss.ids == ["20136", "20137", "20151", "20282"]
    obss.insert(3, obs3)
    assert obss.ids == ["20136", "20137", "20151", "20275", "20282"]
    obss.extend([obs])
    assert obss.ids == ["20136", "20137", "20151", "20275", "20282", "20282"]
    obss.remove(obs)
    assert obss.ids == ["20136", "20137", "20151", "20275", "20282"]
    obss[0] = obs
    assert obss.ids == ["20282", "20137", "20151", "20275", "20282"]

    with pytest.raises(TypeError):
        obss.insert(5, "bad")

    with pytest.raises(TypeError):
        obss[5] = "bad"

    with pytest.raises(TypeError):
        obss[["1", "2"]]


@requires_data()
def test_observations_str(data_store):
    obs_ids = data_store.obs_table["OBS_ID"][:4]
    obss = data_store.get_observations(obs_ids)
    actual = obss.__str__()

    assert actual.split("\n")[1] == "Number of observations: 4"


@requires_data()
def test_observations_select_time_time_intervals_list(data_store):
    obs_ids = data_store.obs_table["OBS_ID"][:8]
    obss = data_store.get_observations(obs_ids)
    # third time interval is out of the observations time range
    time_intervals = [
        Time([53090.130, 53090.140], format="mjd", scale="tt"),
        Time([53110.011, 53110.019], format="mjd", scale="tt"),
        Time([53112.345, 53112.42], format="mjd", scale="tt"),
    ]
    new_obss = obss.select_time(time_intervals)

    assert len(new_obss) == 2
    assert new_obss[0].events.time[0] >= time_intervals[0][0]
    assert new_obss[0].events.time[-1] < time_intervals[0][1]
    assert new_obss[1].events.time[0] >= time_intervals[1][0]
    assert new_obss[1].events.time[-1] < time_intervals[1][1]
    assert_time_allclose(new_obss[0].gti.time_start[0], time_intervals[0][0])
    assert_time_allclose(new_obss[0].gti.time_stop[-1], time_intervals[0][1])
    assert_time_allclose(new_obss[1].gti.time_start[0], time_intervals[1][0])
    assert_time_allclose(new_obss[1].gti.time_stop[-1], time_intervals[1][1])


@requires_data()
def test_observation():
    livetime = 5.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    obs = Observation.create(
        pointing, livetime=livetime, irfs=irfs, deadtime_fraction=0.1
    )

    assert_skycoord_allclose(obs.pointing_radec, pointing.icrs)
    assert_allclose(obs.observation_live_time_duration, 0.9 * livetime)
    assert_allclose(obs.target_radec.ra, np.nan)
    assert_allclose(obs.pointing_zen, np.nan)
    assert_allclose(obs.muoneff, 1)


@requires_data()
class TestObservationChecker:
    def setup(self):
        self.data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")

    def test_check_all(self):
        observation = self.data_store.obs(111140)
        records = list(observation.check())
        assert len(records) == 8

    def test_checker_bad(self):
        for index in range(5):
            self.data_store.hdu_table[index]["FILE_NAME"] = "bad"

        observation = self.data_store.obs(110380)
        records = list(observation.check())
        assert len(records) == 10
        assert records[1]["msg"] == "Loading events failed"
        assert records[3]["msg"] == "Loading GTI failed"
        assert records[5]["msg"] == "Loading aeff failed"
        assert records[7]["msg"] == "Loading edisp failed"
        assert records[9]["msg"] == "Loading psf failed"

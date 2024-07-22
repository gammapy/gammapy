# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from gammapy.data import (
    DataStore,
    EventList,
    Observation,
    ObservationFilter,
    Observations,
)
from gammapy.data.metadata import ObservationMetaData
from gammapy.data.pointing import FixedPointingInfo
from gammapy.data.utils import get_irfs_features
from gammapy.irf import PSF3D, load_irf_dict_from_file
from gammapy.utils.cluster import hierarchical_clustering
from gammapy.utils.fits import HDULocation
from gammapy.utils.testing import (
    assert_skycoord_allclose,
    assert_time_allclose,
    mpl_plot_check,
    requires_data,
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
    assert_skycoord_allclose(obs.get_pointing_icrs(obs.tmid), c)

    c = SkyCoord(22.558341, 41.950807, unit="deg")
    assert_skycoord_allclose(obs.get_pointing_altaz(obs.tmid), c)

    c = SkyCoord(83.63333129882812, 22.01444435119629, unit="deg")
    assert_skycoord_allclose(obs.target_radec, c)


@requires_data()
def test_observation_peek(data_store):
    obs = Observation.read(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )

    with mpl_plot_check():
        obs.peek()

    obs_with_radmax = Observation.read(
        "$GAMMAPY_DATA/magic/rad_max/data/20131004_05029747_DL3_CrabNebula-W0.40+035.fits"
    )

    with mpl_plot_check():
        obs_with_radmax.peek()


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


def test_empty_observations():
    observations = Observations()
    assert len(observations) == 0


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
def test_observation_cta_1dc():
    ontime = 5.0 * u.hr
    pointing = FixedPointingInfo(
        fixed_icrs=SkyCoord(0, 0, unit="deg", frame="galactic").icrs,
    )
    irfs = load_irf_dict_from_file(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    t_ref = Time("2020-01-01T00:00:00")
    tstart = 20 * u.hour
    location = EarthLocation(lon="-70d18m58.84s", lat="-24d41m0.34s", height="2000m")

    obs = Observation.create(
        pointing,
        irfs=irfs,
        deadtime_fraction=0.1,
        tstart=tstart,
        tstop=tstart + ontime,
        reference_time=t_ref,
        location=location,
    )

    assert_skycoord_allclose(obs.get_pointing_icrs(obs.tmid), pointing.fixed_icrs)
    assert_allclose(obs.observation_live_time_duration, 0.9 * ontime)
    assert obs.target_radec is None

    assert isinstance(obs.meta, ObservationMetaData)
    assert obs.meta.deadtime_fraction == 0.1
    assert_allclose(obs.meta.location.height.to_value("m"), 2000)
    assert "Gammapy" in obs.meta.creation.creator


@requires_data()
def test_observation_create_radmax():
    pointing = FixedPointingInfo(
        fixed_icrs=SkyCoord(0, 0, unit="deg", frame="galactic").icrs,
    )
    obs = Observation.read("$GAMMAPY_DATA/joint-crab/dl3/magic/run_05029748_DL3.fits")
    livetime = 5.0 * u.hr
    deadtime = 0.5

    irfs = {}
    for _ in obs.available_irfs:
        irfs[_] = getattr(obs, _)

    obs1 = Observation.create(
        pointing,
        irfs=irfs,
        deadtime_fraction=deadtime,
        livetime=livetime,
    )

    assert_skycoord_allclose(obs1.get_pointing_icrs(obs1.tmid), pointing.fixed_icrs)
    assert_allclose(obs1.observation_live_time_duration, 0.5 * livetime)
    assert obs1.rad_max is not None
    assert obs1.psf is None


@requires_data()
def test_observation_read():
    """read event list and irf components from different DL3 files"""
    obs = Observation.read(
        event_file="$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz",
        irf_file="$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020137.fits.gz",
    )

    energy = Quantity(1, "TeV")
    offset = Quantity(0.5, "deg")
    val = obs.aeff.evaluate(energy_true=energy, offset=offset)

    assert obs.obs_id == 20136
    assert len(obs.events.energy) == 11243
    assert obs.available_hdus == ["events", "gti", "aeff", "edisp", "psf", "bkg"]
    assert_allclose(val.value, 278000.54120855, rtol=1e-5)
    assert val.unit == "m2"

    assert isinstance(obs.meta, ObservationMetaData)
    assert "Gammapy" in obs.meta.creation.creator

    assert obs.meta.obs_info.telescope == "HESS"
    assert obs.meta.obs_info.instrument == "H.E.S.S. Phase I"
    assert obs.meta.target.name == "MSH15-52"
    assert obs.meta.optional["N_TELS"] == 4
    with pytest.raises(KeyError):
        obs.meta.optional["BROKPIX"]


@requires_data()
def test_observation_read_single_file():
    """read event list and irf components from the same DL3 files"""
    obs = Observation.read(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    )

    energy = Quantity(1, "TeV")
    offset = Quantity(0.5, "deg")
    val = obs.aeff.evaluate(energy_true=energy, offset=offset)

    assert obs.obs_id == 20136
    assert len(obs.events.energy) == 11243
    assert obs.available_hdus == ["events", "gti", "aeff", "edisp", "psf", "bkg"]
    assert_allclose(val.value, 273372.44851054, rtol=1e-5)
    assert val.unit == "m2"


@requires_data()
def test_observation_read_single_file_fixed_rad_max():
    """check that for a point-like observation without the RAD_MAX_2D table
    a RadMax2D object is generated from the RAD_MAX keyword"""
    obs = Observation.read("$GAMMAPY_DATA/joint-crab/dl3/magic/run_05029748_DL3.fits")

    assert obs.rad_max is not None
    assert obs.rad_max.quantity.shape == (1, 1)
    assert u.allclose(obs.rad_max.quantity, 0.1414213 * u.deg)


@requires_data()
class TestObservationChecker:
    def setup_method(self):
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


@requires_data()
def test_observation_write(tmp_path):
    obs = Observation.read(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )
    mjdreff = obs.events.table.meta["MJDREFF"]
    mjdrefi = obs.events.table.meta["MJDREFI"]
    path = tmp_path / "obs.fits.gz"

    obs.meta.creation.origin = "test"
    obs.write(path)
    obs_read = obs.read(path)

    assert obs_read.events is not None
    assert obs_read.gti is not None
    assert obs_read.aeff is not None
    assert obs_read.edisp is not None
    assert obs_read.bkg is not None
    assert obs_read.rad_max is None
    assert obs_read.obs_id == 23523
    assert_allclose(obs_read.observatory_earth_location.lat.deg, -23.271778)

    assert_allclose(obs_read.events.table.meta["MJDREFF"], mjdreff)
    assert_allclose(obs_read.events.table.meta["MJDREFI"], mjdrefi)

    # unsupported format
    with pytest.raises(ValueError):
        obs.write(tmp_path / "foo.fits.gz", format="cool-new-format")

    # no irfs
    path = tmp_path / "obs_no_irfs.fits.gz"
    obs.write(path, include_irfs=False)
    obs_read = obs.read(path)
    assert obs_read.events is not None
    assert obs_read.gti is not None
    assert obs_read.aeff is None
    assert obs_read.edisp is None
    assert obs_read.bkg is None
    assert obs_read.rad_max is None


@requires_data()
def test_observation_read_write_checksum(tmp_path):
    obs = Observation.read(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )
    path = tmp_path / "obs.fits"

    obs.write(path, checksum=True)

    with fits.open(path) as hdul:
        for hdu in hdul:
            assert "CHECKSUM" in hdu.header
            assert "DATASUM" in hdu.header

    with open(path, "r+b") as file:
        chunk = file.read(10000)
        index = chunk.find("EV_CLASS".encode("ascii"))
        file.seek(index)
        file.write("BAD__KEY".encode("ascii"))

    with pytest.warns(UserWarning):
        Observation.read(path, checksum=True)


@requires_data()
def test_obervation_copy(data_store):
    obs = data_store.obs(23523)

    obs_copy = obs.copy()
    assert obs_copy.obs_id == 23523
    assert isinstance(obs_copy.__dict__["_psf_hdu"], HDULocation)

    with pytest.raises(ValueError):
        _ = obs.copy(obs_id=1234)

    obs_copy = obs.copy(obs_id=1234, in_memory=True)
    assert isinstance(obs_copy.__dict__["psf"], PSF3D)
    assert obs_copy.obs_id == 1234


def test_observation_tmid():
    from gammapy.data import GTI

    start = Time("2020-01-01T20:00:00")
    stop = Time("2020-01-01T20:10:00")
    expected = Time("2020-01-01T20:05:00")
    epoch = Time("2020-01-01T00:00:00")

    gti = GTI.create([(start - epoch).to(u.s)], [(stop - epoch).to(u.s)], epoch)
    obs = Observation(gti=gti)
    assert abs(obs.tmid - expected).to(u.ns) < 1 * u.us


@requires_data()
def test_observations_clustering(data_store):
    selection = dict(
        type="sky_circle",
        frame="icrs",
        lon="83.633 deg",
        lat="22.014 deg",
        radius="2 deg",
    )
    obs_table = data_store.obs_table.select_observations(selection)
    observations = data_store.get_observations(obs_table["OBS_ID"])

    coord = SkyCoord(83.63308, 22.01450, unit="deg", frame="icrs")
    names = ["edisp-bias", "edisp-res", "psf-radius"]
    features = get_irfs_features(
        observations, energy_true="1 TeV", position=coord, names=names
    )

    n_features = len(names)
    features_array = np.array(
        [
            features[col].data
            for col in features.columns
            if col not in ["obs_id", "dataset_name"]
        ]
    ).T
    assert features_array.shape == (len(observations), n_features)

    features = hierarchical_clustering(
        features, linkage_kwargs={"method": "complete"}, fcluster_kwargs={"t": 2}
    )

    assert np.all(
        features["labels"].data
        == np.array(
            [
                1,
                1,
                2,
                2,
            ]
        )
    )

    features = get_irfs_features(
        observations,
        energy_true="1 TeV",
        position=coord,
        names=names,
        apply_standard_scaler=True,
    )
    features = hierarchical_clustering(features)
    features_array = np.array(
        [
            features[col].data
            for col in features.columns
            if col not in ["obs_id", "dataset_name"]
        ]
    ).T

    obs_clusters = observations.group_by_label(features["labels"])
    for k in range(n_features):
        assert_allclose(features_array[:, k].mean(), 0, atol=1e-7)
        assert_allclose(features_array[:, k].std(), 1, atol=1e-7)

    assert np.all(features["labels"].data == np.array([2, 1, 1, 1]))

    assert len(obs_clusters["group_1"]) == 3
    assert len(obs_clusters["group_2"]) == 1
    assert obs_clusters["group_2"][0].obs_id == 23523


@requires_data()
def test_filter_live_time_phase(data_store):
    observation = data_store.obs(20136)
    phase_filter = {"type": "custom", "opts": dict(parameter="PHASE", band=(0.2, 0.8))}

    default_obs_live_time = observation.observation_live_time_duration

    obs_filter = ObservationFilter(event_filters=[phase_filter])
    observation.obs_filter = obs_filter
    live_time_filter = observation.observation_live_time_duration

    assert_allclose(live_time_filter, default_obs_live_time * (0.8 - 0.2))


@requires_data()
def test_stack_observations(data_store, caplog):
    obs_1 = data_store.get_observations([20136, 20137, 20151])
    obs_2 = data_store.get_observations([20275, 20282])

    obs12 = Observations.from_stack([obs_1, obs_2])

    assert len(obs12) == 5
    assert isinstance(obs12[0], Observation)

    obs122 = Observations.from_stack([obs12, obs_2])

    assert len(obs122) == 7
    assert "WARNING" in [_.levelname for _ in caplog.records]
    assert "Observation with obs_id 20275 already belongs to Observations." in [
        _.message for _ in caplog.records
    ]

    caplog.clear()
    obs_1[2] = obs_1[0]

    assert "WARNING" in [_.levelname for _ in caplog.records]
    assert "Observation with obs_id 20136 already belongs to Observations." in [
        _.message for _ in caplog.records
    ]

    with pytest.raises(TypeError):
        Observations.from_stack([obs_1, ["a"]])


@requires_data()
def test_observations_generator(data_store):
    """Test Observations.generator()"""
    obs_1 = data_store.get_observations([20136, 20137, 20151])

    for idx, obs in enumerate(obs_1.in_memory_generator()):
        assert isinstance(obs, Observation)
        assert obs.obs_id == obs_1[idx].obs_id
        assert isinstance(obs.events, EventList)
        assert isinstance(obs.psf, PSF3D)


@requires_data()
def test_event_setter():
    irfs = load_irf_dict_from_file(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    pointing = FixedPointingInfo(
        fixed_icrs=SkyCoord(0 * u.deg, 0 * u.deg),
    )
    location = EarthLocation(lon="-70d18m58.84s", lat="-24d41m0.34s", height="2000m")
    obs = Observation.create(
        obs_id=1,
        pointing=pointing,
        livetime=20 * u.min,
        irfs=irfs,
        location=location,
    )

    assert obs.events is None

    for invalid in (5, Table(), "foo"):
        with pytest.raises(TypeError):
            obs.events = invalid

    events = EventList(Table())
    obs.events = events
    assert obs.events is events


@requires_data()
def test_observations_getitem(data_store):
    """Test mask indexing on Observations"""
    obs_1 = data_store.get_observations([20136, 20137, 20151])

    assert isinstance(obs_1[0], Observation)
    assert isinstance(obs_1[1:], Observations)
    assert len(obs_1[1:]) == 2

    mask = [True, False, True]
    obs_2 = obs_1[mask]

    assert len(obs_2) == 2
    assert obs_2.ids == ["20136", "20151"]

    ind = [0, 2]
    obs_2 = obs_1[ind]

    assert len(obs_2) == 2
    assert obs_2.ids == ["20136", "20151"]

    obs_2 = obs_1[np.array(mask)]

    assert len(obs_2) == 2
    assert obs_2.ids == ["20136", "20151"]

    assert obs_1[["20136", "20151"]].ids == ["20136", "20151"]

    obs_2 = obs_1[[]]
    assert len(obs_2) == 0
    assert isinstance(obs_2, Observations)

    obs_2 = obs_1[np.array([])]
    assert len(obs_2) == 0
    assert isinstance(obs_2, Observations)

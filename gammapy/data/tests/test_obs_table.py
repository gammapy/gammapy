# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.time import Time, TimeDelta
from astropy.units import Quantity
from gammapy.data import GTI, observatory_locations
from gammapy.data.obs_table import ObservationTable, ObservationTableChecker
from gammapy.utils.random import get_random_state, sample_sphere
from gammapy.utils.testing import (
    assert_quantity_allclose,
    assert_time_allclose,
    requires_data,
)
from gammapy.utils.time import time_ref_from_dict, time_relative_to_ref


def make_test_observation_table(
    observatory_name="hess",
    n_obs=10,
    az_range=Angle([0, 360], "deg"),
    alt_range=Angle([45, 90], "deg"),
    date_range=(Time("2010-01-01"), Time("2015-01-01")),
    use_abs_time=False,
    n_tels_range=(3, 4),
    random_state="random-seed",
):
    """Make a test observation table.
    Create an observation table following a specific pattern.
    For the moment, only random observation tables are created.
    The observation table is created according to a specific
    observatory, and randomizing the observation pointingpositions
    in a specified az-alt range.
    If a *date_range* is specified, the starting time
    of the observations will be restricted to the specified interval.
    These parameters are interpreted as date, the precise hour of the
    day is ignored, unless the end date is closer than 1 day to the
    starting date, in which case, the precise time of the day is also
    considered.
    In addition, a range can be specified for the number of telescopes.
    Parameters
    ----------
    observatory_name : str, optional
        Name of the observatory; a list of choices is given in
        `~gammapy.data.observatory_locations`.
    n_obs : int, optional
        Number of observations for the obs table.
    az_range : `~astropy.coordinates.Angle`, optional
        Azimuth angle range (start, end) for random generation of
        observation pointing positions.
    alt_range : `~astropy.coordinates.Angle`, optional
        Altitude angle range (start, end) for random generation of
        observation pointing positions.
    date_range : `~astropy.time.Time`, optional
        Date range (start, end) for random generation of observation
        start time.
    use_abs_time : bool, optional
        Use absolute UTC times instead of [MET]_ seconds after the reference.
    n_tels_range : int, optional
        Range (start, end) of number of telescopes participating in
        the observations.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}, optional
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.
    Returns
    -------
    obs_table : `~gammapy.data.ObservationTable`
        Observation table.
    """
    random_state = get_random_state(random_state)

    n_obs_start = 1

    obs_table = ObservationTable()

    # build a time reference as the start of 2010
    dateref = Time("2010-01-01T00:00:00")
    dateref_mjd_fra, dateref_mjd_int = np.modf(dateref.mjd)

    # define table header
    obs_table.meta["OBSERVATORY_NAME"] = observatory_name
    obs_table.meta["MJDREFI"] = dateref_mjd_int
    obs_table.meta["MJDREFF"] = dateref_mjd_fra
    obs_table.meta["TIMESYS"] = "TT"
    obs_table.meta["TIMEUNIT"] = "s"
    obs_table.meta["TIMEREF"] = "LOCAL"
    if use_abs_time:
        # show the observation times in UTC
        obs_table.meta["TIME_FORMAT"] = "absolute"
    else:
        # show the observation times in seconds after the reference
        obs_table.meta["TIME_FORMAT"] = "relative"
    header = obs_table.meta

    # obs id
    obs_id = np.arange(n_obs_start, n_obs_start + n_obs)
    obs_table["OBS_ID"] = obs_id

    # obs time: 30 min
    ontime = Quantity(30.0 * np.ones_like(obs_id), "minute").to("second")
    obs_table["ONTIME"] = ontime

    # livetime: 25 min
    time_live = Quantity(25.0 * np.ones_like(obs_id), "minute").to("second")
    obs_table["LIVETIME"] = time_live

    # start time
    #  - random points between the start of 2010 and the end of 2014 (unless
    # otherwise specified)
    #  - using the start of 2010 as a reference time for the header of the table
    #  - observations restrict to night time (only if specified time interval is
    # more than 1 day)
    #  - considering start of astronomical day at midday: implicit in setting
    # the start of the night, when generating random night hours
    datestart = date_range[0]
    dateend = date_range[1]
    time_start = random_state.uniform(datestart.mjd, dateend.mjd, len(obs_id))
    time_start = Time(time_start, format="mjd", scale="utc")

    # check if time interval selected is more than 1 day
    if (dateend - datestart).jd > 1.0:
        # keep only the integer part (i.e. the day, not the fraction of the day)
        time_start_f, time_start_i = np.modf(time_start.mjd)
        time_start = Time(time_start_i, format="mjd", scale="utc")

        # random generation of night hours: 6 h (from 22 h to 4 h), leaving 1/2 h
        # time for the last run to finish
        night_start = Quantity(22.0, "hour")
        night_duration = Quantity(5.5, "hour")
        hour_start = random_state.uniform(
            night_start.value, night_start.value + night_duration.value, len(obs_id)
        )
        hour_start = Quantity(hour_start, "hour")

        # add night hour to integer part of MJD
        time_start += hour_start

    if use_abs_time:
        # show the observation times in UTC
        time_start = Time(time_start.isot)
    else:
        # show the observation times in seconds after the reference
        time_start = time_relative_to_ref(time_start, header)
        # converting to quantity (better treatment of units)
        time_start = Quantity(time_start.sec, "second")

    obs_table["TSTART"] = time_start

    # stop time
    # calculated as TSTART + ONTIME
    if use_abs_time:
        time_stop = Time(obs_table["TSTART"])
        time_stop += TimeDelta(obs_table["ONTIME"])
    else:
        time_stop = TimeDelta(obs_table["TSTART"])
        time_stop += TimeDelta(obs_table["ONTIME"])
        # converting to quantity (better treatment of units)
        time_stop = Quantity(time_stop.sec, "second")

    obs_table["TSTOP"] = time_stop

    # az, alt
    # random points in a portion of sphere; default: above 45 deg altitude
    az, alt = sample_sphere(
        size=len(obs_id),
        lon_range=az_range,
        lat_range=alt_range,
        random_state=random_state,
    )
    az = Angle(az, "deg")
    alt = Angle(alt, "deg")
    obs_table["AZ"] = az
    obs_table["ALT"] = alt

    # RA, dec
    # derive from az, alt taking into account that alt, az represent the values
    # at the middle of the observation, i.e. at time_ref + (TIME_START + TIME_STOP)/2
    # (or better: time_ref + TIME_START + (TIME_OBSERVATION/2))
    # in use_abs_time mode, the time_ref should not be added, since it's already included
    # in TIME_START and TIME_STOP
    az = Angle(obs_table["AZ"])
    alt = Angle(obs_table["ALT"])
    if use_abs_time:
        obstime = Time(obs_table["TSTART"])
        obstime += TimeDelta(obs_table["ONTIME"]) / 2.0
    else:
        obstime = time_ref_from_dict(obs_table.meta)
        obstime += TimeDelta(obs_table["TSTART"])
        obstime += TimeDelta(obs_table["ONTIME"]) / 2.0
    location = observatory_locations[observatory_name]
    altaz_frame = AltAz(obstime=obstime, location=location)
    alt_az_coord = SkyCoord(az, alt, frame=altaz_frame)
    sky_coord = alt_az_coord.transform_to("icrs")
    obs_table["RA_PNT"] = sky_coord.ra
    obs_table["DEC_PNT"] = sky_coord.dec

    # positions

    # number of telescopes
    # random integers in a specified range; default: between 3 and 4
    n_tels = random_state.randint(n_tels_range[0], n_tels_range[1] + 1, len(obs_id))
    obs_table["N_TELS"] = n_tels

    # muon efficiency
    # random between 0.6 and 1.0
    muoneff = random_state.uniform(low=0.6, high=1.0, size=len(obs_id))
    obs_table["MUONEFF"] = muoneff

    return obs_table


def test_basics():
    random_state = np.random.RandomState(seed=0)
    obs_table = make_test_observation_table(n_obs=10, random_state=random_state)
    assert obs_table.summary().startswith("Observation table")


def test_select_parameter_box():
    # create random observation table
    random_state = np.random.RandomState(seed=0)
    obs_table = make_test_observation_table(n_obs=10, random_state=random_state)

    # select some pars and check the corresponding values in the columns

    # test box selection in obs_id
    variable = "OBS_ID"
    value_range = [2, 5]
    selection = dict(type="par_box", variable=variable, value_range=value_range)
    selected_obs_table = obs_table.select_observations(selection)
    assert len(selected_obs_table) == 3
    assert (value_range[0] <= selected_obs_table[variable]).all()
    assert (selected_obs_table[variable] < value_range[1]).all()

    # test box selection in obs_id inverted
    selection = dict(
        type="par_box", variable=variable, value_range=value_range, inverted=True
    )
    selected_obs_table = obs_table.select_observations(selection)
    assert len(selected_obs_table) == 7
    assert (
        (value_range[0] > selected_obs_table[variable])
        | (selected_obs_table[variable] >= value_range[1])
    ).all()

    # test box selection in alt
    variable = "ALT"
    value_range = Angle([60.0, 70.0], "deg")
    selection = dict(type="par_box", variable=variable, value_range=value_range)
    selected_obs_table = obs_table.select_observations(selection)
    assert (value_range[0] < Angle(selected_obs_table[variable])).all()
    assert (Angle(selected_obs_table[variable]) < value_range[1]).all()


def test_select_time_box():
    # create random observation table with very close (in time)
    # observations (and times in absolute times)
    datestart = Time("2012-01-01T00:30:00")
    dateend = Time("2012-01-01T02:30:00")
    random_state = np.random.RandomState(seed=0)
    obs_table_time = make_test_observation_table(
        n_obs=10,
        date_range=(datestart, dateend),
        use_abs_time=True,
        random_state=random_state,
    )

    # test box selection in time: (time_start, time_stop) within value_range
    value_range = Time(["2012-01-01T01:00:00", "2012-01-01T02:00:00"])
    selection = dict(type="time_box", time_range=value_range)
    selected_obs_table = obs_table_time.select_observations(selection)
    time_start = selected_obs_table["TSTART"]
    assert (value_range[0] < time_start).all()
    assert (time_start < value_range[1]).all()


def test_select_sky_regions():
    random_state = np.random.RandomState(seed=0)
    obs_table = make_test_observation_table(n_obs=100, random_state=random_state)

    selection = dict(
        type="sky_circle",
        frame="galactic",
        lon="0 deg",
        lat="0 deg",
        radius="50 deg",
        border="2 deg",
    )
    obs_table = obs_table.select_observations(selection)
    assert len(obs_table) == 32


def test_create_gti():
    date_start = Time("2012-01-01T00:30:00")
    date_end = Time("2012-01-01T02:30:00")
    random_state = np.random.RandomState(seed=0)
    obs_table = make_test_observation_table(
        n_obs=1, date_range=(date_start, date_end), random_state=random_state
    )

    gti = obs_table.create_gti(obs_id=1)

    met_ref = time_ref_from_dict(obs_table.meta)
    time_start = met_ref + Quantity(obs_table[0]["TSTART"].astype("float64"), "second")
    time_stop = met_ref + Quantity(obs_table[0]["TSTOP"].astype("float64"), "second")

    assert isinstance(gti, GTI)
    assert_time_allclose(gti.time_start, time_start)
    assert_time_allclose(gti.time_stop, time_stop)
    assert_quantity_allclose(
        gti.time_sum, Quantity(obs_table[0]["ONTIME"].astype("float64"), "second")
    )


@requires_data()
def test_observation_table_checker():
    path = "$GAMMAPY_DATA/cta-1dc/index/gps/obs-index.fits.gz"
    obs_table = ObservationTable.read(path)
    checker = ObservationTableChecker(obs_table)

    records = list(checker.run())
    assert len(records) == 1

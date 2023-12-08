# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import QTable, Table
from astropy.time import Time
from gammapy.data import GTI
from gammapy.utils.testing import assert_time_allclose, requires_data


def make_gti(mets, time_ref="2010-01-01"):
    """Create GTI from a dict of MET (assumed to be in seconds) and a reference time."""
    time_ref = Time(time_ref)
    times = {name: time_ref + met for name, met in mets.items()}
    table = Table(times)
    return GTI(table, reference_time=time_ref)


def test_gti_table_validation():
    start = Time([53090.123451203704], format="mjd", scale="tt")
    stop = Time([53090.14291879629], format="mjd", scale="tt")

    table = QTable([start, stop], names=["START", "STOP"])
    validated = GTI._validate_table(table)
    assert validated == table

    bad_table = QTable([start, stop], names=["bad", "STOP"])
    with pytest.raises(ValueError):
        GTI._validate_table(bad_table)

    with pytest.raises(TypeError):
        GTI._validate_table([start, stop])

    bad_table = QTable([[1], [2]], names=["START", "STOP"])
    with pytest.raises(TypeError):
        GTI._validate_table(bad_table)


def test_simple_gti():
    time_ref = Time("2010-01-01")
    gti = make_gti({"START": [0, 2] * u.s, "STOP": [1, 3] * u.s}, time_ref=time_ref)

    assert_allclose(gti.time_start.mjd - time_ref.mjd, [0, 2.3148146e-5])
    assert_allclose(
        (gti.time_stop - time_ref).to_value("d"), [1.15740741e-05, 3.4722222e-05]
    )
    assert_allclose(gti.time_delta.to_value("s"), [1, 1])
    assert_allclose(gti.time_sum.to_value("s"), 2.0)


@requires_data()
def test_gti_hess():
    gti = GTI.read("$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz")

    str_gti = str(gti)
    assert "Start: 101962602.0 s MET" in str_gti
    assert "Stop: 2004-03-26T03:25:48.184 (time standard: TT)" in str_gti
    assert "GTI" in str_gti

    assert len(gti.table) == 1

    assert gti.time_delta[0].unit == "s"
    assert_allclose(gti.time_delta[0].value, 1682)
    assert_allclose(gti.time_sum.value, 1682)

    expected = Time(53090.123451203704, format="mjd", scale="tt")
    assert_time_allclose(gti.time_start[0], expected)

    expected = Time(53090.14291879629, format="mjd", scale="tt")
    assert_time_allclose(gti.time_stop[0], expected)


@requires_data()
def test_gti_fermi():
    gti = GTI.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz")
    assert "GTI" in str(gti)
    assert len(gti.table) == 39042

    assert gti.time_delta[0].unit == "s"
    assert_allclose(gti.time_delta[0].value, 651.598893)
    assert_allclose(gti.time_sum.value, 1.831396e08)

    expected = Time(54682.65603794185, format="mjd", scale="tt")
    assert_time_allclose(gti.time_start[0], expected)

    expected = Time(54682.66357959571, format="mjd", scale="tt")
    assert_time_allclose(gti.time_stop[0], expected)


@requires_data()
@pytest.mark.parametrize(
    "time_interval, expected_length, expected_times",
    [
        (
            Time(
                ["2008-08-04T16:21:00", "2008-08-04T19:10:00"],
                format="isot",
                scale="tt",
            ),
            2,
            Time(
                ["2008-08-04T16:21:00", "2008-08-04T19:10:00"],
                format="isot",
                scale="tt",
            ),
        ),
        (
            Time([54682.68125, 54682.79861111], format="mjd", scale="tt"),
            2,
            Time([54682.68125, 54682.79861111], format="mjd", scale="tt"),
        ),
        (
            Time([10.0, 100000.0], format="mjd", scale="tt"),
            39042,
            Time([54682.65603794185, 57236.96833546296], format="mjd", scale="tt"),
        ),
        (Time([10.0, 20.0], format="mjd", scale="tt"), 0, None),
    ],
)
def test_select_time(time_interval, expected_length, expected_times):
    gti = GTI.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz")

    gti_selected = gti.select_time(time_interval)

    assert len(gti_selected.table) == expected_length

    if expected_length != 0:
        expected_times.format = "mjd"
        assert_time_allclose(gti_selected.time_start[0], expected_times[0])
        assert_time_allclose(gti_selected.time_stop[-1], expected_times[1])


def test_gti_delete_intervals():
    gti = GTI.create(
        Time(
            [
                "2020-01-01 01:00:00.000",
                "2020-01-01 02:00:00.000",
                "2020-01-01 03:00:00.000",
                "2020-01-01 05:00:00.000",
            ]
        ),
        Time(
            [
                "2020-01-01 01:45:00.000",
                "2020-01-01 02:45:00.000",
                "2020-01-01 03:45:00.000",
                "2020-01-01 05:45:00.000",
            ]
        ),
    )
    interval = Time(["2020-01-01 02:20:00.000", "2020-01-01 05:20:00.000"])
    interval2 = Time(["2020-01-01 05:30:00.000", "2020-01-01 08:20:00.000"])
    interval3 = Time(["2020-01-01 05:50:00.000", "2020-01-01 09:20:00.000"])

    gti_trim = gti.delete_interval(interval)

    assert len(gti_trim.table) == 3

    assert_time_allclose(
        gti_trim.table["START"],
        Time(
            [
                "2020-01-01 01:00:00.000",
                "2020-01-01 02:00:00.000",
                "2020-01-01 05:20:00.000",
            ]
        ),
    )
    assert_time_allclose(
        gti_trim.table["STOP"],
        Time(
            [
                "2020-01-01 01:45:00.000",
                "2020-01-01 02:20:00.000",
                "2020-01-01 05:45:00.000",
            ]
        ),
    )

    gti_trim2 = gti_trim.delete_interval(interval2)
    assert_time_allclose(
        gti_trim2.table["STOP"],
        Time(
            [
                "2020-01-01 01:45:00.000",
                "2020-01-01 02:20:00.000",
                "2020-01-01 05:30:00.000",
            ]
        ),
    )

    gti_trim3 = gti_trim2.delete_interval(interval3)
    assert_time_allclose(
        gti_trim3.table["STOP"],
        Time(
            [
                "2020-01-01 01:45:00.000",
                "2020-01-01 02:20:00.000",
                "2020-01-01 05:30:00.000",
            ]
        ),
    )


def test_gti_stack():
    time_ref = Time("2010-01-01")
    gti1 = make_gti({"START": [0, 2] * u.s, "STOP": [1, 3] * u.s}, time_ref=time_ref)
    gt1_pre_stack = gti1.copy()
    gti2 = make_gti(
        {"START": [4] * u.s, "STOP": [5] * u.s}, time_ref=time_ref + 10 * u.s
    )

    gti1.stack(gti2)

    assert len(gti1.table) == 3
    assert_time_allclose(gt1_pre_stack.time_ref, gti1.time_ref)

    assert_allclose(gti1.met_start.value, [0, 2, 14])
    assert_allclose(gti1.met_stop.value, [1, 3, 15])


def test_gti_union():
    gti = make_gti({"START": [5, 6, 1, 2] * u.s, "STOP": [8, 7, 3, 4] * u.s})

    gti = gti.union()

    assert_allclose(gti.met_start.value, [1, 5])
    assert_allclose(gti.met_stop.value, [4, 8])


def test_gti_create():
    start = u.Quantity([1, 2], "min")
    stop = u.Quantity([1.5, 2.5], "min")
    time_ref = Time("2010-01-01 00:00:00.0")

    gti = GTI.create(start, stop, time_ref)

    assert len(gti.table) == 2
    assert_allclose(gti.time_ref.mjd, time_ref.tt.mjd)
    start_met = (gti.time_start - gti.time_ref).to_value("s")
    assert_allclose(start_met, start.to_value("s"))


def test_gti_write(tmp_path):
    time_ref = Time("2010-01-01", scale="tt")
    time_ref.format = "mjd"
    gti = make_gti({"START": [5, 6, 1, 2] * u.s, "STOP": [8, 7, 3, 4] * u.s}, time_ref)

    gti.write(tmp_path / "tmp.fits")
    new_gti = GTI.read(tmp_path / "tmp.fits")

    assert_time_allclose(new_gti.time_start, gti.time_start)
    assert_time_allclose(new_gti.time_stop, gti.time_stop)
    assert_time_allclose(new_gti.time_ref, gti.time_ref)


def test_gti_from_time():
    """Test astropy time is supported as input for GTI.create"""
    start = Time("2020-01-01T20:00:00")
    stop = Time("2020-01-01T20:15:00")
    ref = Time("2020-01-01T00:00:00")
    gti = GTI.create(start, stop, ref)

    assert_time_allclose(gti.table["START"], start)
    assert_time_allclose(gti.table["STOP"], stop)

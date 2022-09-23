# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.data import GTI
from gammapy.utils.testing import assert_time_allclose, requires_data
from gammapy.utils.time import time_ref_to_dict


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


def make_gti(times, time_ref="2010-01-01"):
    meta = time_ref_to_dict(time_ref)
    table = Table(times, meta=meta)
    return GTI(table)


def test_gti_stack():
    time_ref = Time("2010-01-01")
    gti1 = make_gti({"START": [0, 2], "STOP": [1, 3]}, time_ref=time_ref)
    gt1_pre_stack = gti1.copy()
    gti2 = make_gti({"START": [4], "STOP": [5]}, time_ref=time_ref + 10 * u.s)

    gti1.stack(gti2)

    assert len(gti1.table) == 3
    assert_time_allclose(gt1_pre_stack.time_ref, gti1.time_ref)
    assert_allclose(gti1.table["START"], [0, 2, 14])
    assert_allclose(gti1.table["STOP"], [1, 3, 15])


def test_gti_union():
    gti = make_gti({"START": [5, 6, 1, 2], "STOP": [8, 7, 3, 4]})

    gti = gti.union()

    assert_allclose(gti.table["START"], [1, 5])
    assert_allclose(gti.table["STOP"], [4, 8])


def test_gti_create():
    start = u.Quantity([1, 2], "min")
    stop = u.Quantity([1.5, 2.5], "min")
    time_ref = Time("2010-01-01 00:00:00.0")

    gti = GTI.create(start, stop, time_ref)

    assert len(gti.table) == 2
    assert_allclose(gti.time_ref.mjd, time_ref.tt.mjd)
    assert_allclose(gti.table["START"], start.to_value("s"))


def test_gti_write(tmp_path):
    gti = make_gti({"START": [5, 6, 1, 2], "STOP": [8, 7, 3, 4]})

    gti.write(tmp_path / "tmp.fits")
    new_gti = GTI.read(tmp_path / "tmp.fits")

    assert_time_allclose(new_gti.time_start, gti.time_start)
    assert_time_allclose(new_gti.time_stop, gti.time_stop)
    assert new_gti.table.meta["MJDREFF"] == gti.table.meta["MJDREFF"]


def test_gti_from_time():
    """Test astropy time is supported as input for GTI.create"""
    start = Time("2020-01-01T20:00:00")
    stop = Time("2020-01-01T20:15:00")
    ref = Time("2020-01-01T00:00:00")
    gti = GTI.create(start, stop, ref)

    assert u.isclose(gti.table["START"], 20 * u.hour)
    assert u.isclose(gti.table["STOP"], 20 * u.hour + 15 * u.min)

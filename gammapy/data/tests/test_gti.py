# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.time import Time
import astropy.units as u
from astropy.table import Table
from ...utils.testing import requires_data, assert_time_allclose
from ...data import GTI
from ...utils.time import time_ref_to_dict


@requires_data()
def test_gti_hess():
    filename = "$GAMMAPY_DATA/tests/unbundled/hess/run_0023037_hard_eventlist.fits.gz"
    gti = GTI.read(filename)
    assert "GTI" in str(gti)
    assert len(gti.table) == 1

    assert gti.time_delta[0].unit == "s"
    assert_allclose(gti.time_delta[0].value, 1568.00000)
    assert_allclose(gti.time_sum.value, 1568.00000)

    expected = Time(53292.00592592593, format="mjd", scale="tt")
    assert_time_allclose(gti.time_start[0], expected)

    expected = Time(53292.02407407408, format="mjd", scale="tt")
    assert_time_allclose(gti.time_stop[0], expected)


@requires_data()
def test_gti_fermi():
    filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"
    gti = GTI.read(filename)
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
    filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"
    gti = GTI.read(filename)

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
    gti2 = make_gti({"START": [4], "STOP": [5]}, time_ref=time_ref + 10 * u.s)

    gti = gti1.stack(gti2)

    assert_time_allclose(gti.time_ref, gti1.time_ref)
    assert_allclose(gti.table["START"], [0, 2, 14])
    assert_allclose(gti.table["STOP"], [1, 3, 15])


def test_gti_union():
    gti = make_gti({"START": [5, 6, 1, 2], "STOP": [8, 7, 3, 4]})

    gti = gti.union()

    assert_allclose(gti.table["START"], [1, 5])
    assert_allclose(gti.table["STOP"], [4, 8])

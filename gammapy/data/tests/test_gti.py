# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
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


@pytest.fixture(scope="session")
def ref_dict():
    time_ref = Time("2018-10-29 20:00:00.000")
    return time_ref_to_dict(time_ref)


@pytest.fixture(scope="session")
def ref_dict2():
    time_ref = Time("2018-10-30 20:00:00.000")
    return time_ref_to_dict(time_ref)


@pytest.fixture(scope="session")
def gti_list(ref_dict):
    time_start = 30 * u.min
    time_step = 60 * u.min
    duration = 30 * u.min
    nstep = 3
    start = time_start + time_step * np.arange(nstep)
    stop = start + duration

    gti_table = Table(
        [start.to("s"), stop.to("s")], names=("START", "STOP"), meta=ref_dict
    )
    return GTI(gti_table)


@pytest.fixture(scope="session")
def outside_gti(ref_dict2):
    start = u.Quantity([1 * u.d])
    stop = start + 30 * u.min
    gti_table = Table(
        [start.to("s"), stop.to("s")], names=("START", "STOP"), meta=ref_dict2
    )
    return GTI(gti_table)


@pytest.fixture(scope="session")
def overlapping_gti(ref_dict):
    start = u.Quantity([0 * u.s])
    stop = start + 110 * u.min
    gti_table = Table(
        [start.to("s"), stop.to("s")], names=("START", "STOP"), meta=ref_dict
    )
    return GTI(gti_table)


def test_gti_stack(gti_list, outside_gti):
    stacked_gti = gti_list.stack(outside_gti)
    assert len(stacked_gti.table) == 4
    assert stacked_gti.time_ref == gti_list.time_ref

    inverse_stacked_gti = outside_gti.stack(gti_list)
    assert inverse_stacked_gti.time_ref == outside_gti.time_ref
    assert len(inverse_stacked_gti.table) == 4


def test_gti_union(gti_list, outside_gti, overlapping_gti):
    # interval after all intervals in the list
    union = gti_list.union(outside_gti)
    assert len(union.table) == 4
    assert_allclose(union.time_sum.to_value("h"), 2)
    assert_time_allclose(union.time_start[3], outside_gti.time_start[0])

    # interval covering the first interval in the list and part of the second
    union = gti_list.union(overlapping_gti)
    assert len(union.table) == 2
    assert_allclose(union.time_sum.to_value("h"), 2.5)
    assert_time_allclose(union.time_start[0], overlapping_gti.time_start[0])
    assert_time_allclose(union.time_stop[0], gti_list.time_stop[1])

    # now take union of outside and overlap
    new_gti = overlapping_gti.union(outside_gti)
    union = gti_list.union(new_gti)

    assert len(new_gti.table) == 2
    assert len(union.table) == 3
    assert_allclose(union.time_sum.to_value("h"), 3)
    assert_time_allclose(union.time_start[0], overlapping_gti.time_start[0])
    assert_time_allclose(union.time_stop[0], gti_list.time_stop[1])
    assert_time_allclose(union.time_start[2], outside_gti.time_start[0])
    assert_time_allclose(union.time_ref, gti_list.time_ref)

    # Reverse union order should give the same result
    union2 = new_gti.union(gti_list)
    assert_time_allclose(union.time_start, union2.time_start)
    assert_time_allclose(union.time_stop, union2.time_stop)
    assert_time_allclose(union.time_ref, new_gti.time_ref)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from astropy.time import Time
from ...utils.testing import requires_data, assert_time_allclose
from ...data import GTI


@requires_data("gammapy-extra")
def test_gti_hess():
    filename = (
        "$GAMMAPY_EXTRA/test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz"
    )
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


@requires_data("gammapy-extra")
def test_gti_fermi():
    filename = "$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz"
    gti = GTI.read(filename)
    assert "GTI" in str(gti)
    assert len(gti.table) == 36589

    assert gti.time_delta[0].unit == "s"
    assert_allclose(gti.time_delta[0].value, 352.49307)
    assert_allclose(gti.time_sum.value, 171273490.97510)

    expected = Time(54682.659499814814, format="mjd", scale="tt")
    assert_time_allclose(gti.time_start[0], expected)

    expected = Time(54682.66357959571, format="mjd", scale="tt")
    assert_time_allclose(gti.time_stop[0], expected)


@requires_data("gammapy-extra")
@pytest.mark.parametrize(
    "time_interval",
    [
        Time(["2008-08-04T16:21:00", "2008-08-04T19:10:00"], format="isot", scale="tt"),
        Time([54682.68125, 54682.79861111], format="mjd", scale="tt"),
    ],
)
def test_select_time(time_interval):
    filename = "$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz"
    gti = GTI.read(filename)

    gti_selected = gti.select_time(time_interval)

    assert len(gti_selected.table) == 2

    time_interval.format = "mjd"
    assert_time_allclose(gti_selected.time_start[0], time_interval[0])
    assert_time_allclose(gti_selected.time_stop[-1], time_interval[1])

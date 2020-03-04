# Licensed under a 3-clause BSD style license - see LICENSE.rst
import datetime
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Column, Table
from astropy.time import Time
from gammapy.time import LightCurve
from gammapy.utils.testing import mpl_plot_check, requires_dependency


@pytest.fixture(scope="session")
def lc():
    meta = dict(TIMESYS="utc")

    table = Table(
        meta=meta,
        data=[
            Column(Time(["2010-01-01", "2010-01-03"]).mjd, "time_min"),
            Column(Time(["2010-01-03", "2010-01-10"]).mjd, "time_max"),
            Column([1e-11, 3e-11], "flux", unit="cm-2 s-1"),
            Column([0.1e-11, 0.3e-11], "flux_err", unit="cm-2 s-1"),
            Column([np.nan, 3.6e-11], "flux_ul", unit="cm-2 s-1"),
            Column([False, True], "is_ul"),
        ],
    )

    return LightCurve(table=table)


def test_lightcurve_repr(lc):
    assert repr(lc) == "LightCurve(len=2)"


def test_lightcurve_properties_time(lc):
    assert lc.time_scale == "utc"
    assert lc.time_format == "mjd"

    # Time-related attributes
    time = lc.time
    assert time.scale == "utc"
    assert time.format == "mjd"
    assert_allclose(time.mjd, [55198, 55202.5])

    assert_allclose(lc.time_min.mjd, [55197, 55199])
    assert_allclose(lc.time_max.mjd, [55199, 55206])

    # Note: I'm not sure why the time delta has this scale and format
    time_delta = lc.time_delta
    assert time_delta.scale == "tai"
    assert time_delta.format == "jd"
    assert_allclose(time_delta.jd, [2, 7])


def test_lightcurve_properties_flux(lc):
    flux = lc.table["flux"].quantity
    assert flux.unit == "cm-2 s-1"
    assert_allclose(flux.value, [1e-11, 3e-11])


# TODO: extend these tests to cover other time scales.
# In those cases, CSV should not round-trip because there
# is no header info in CSV to store the time scale!


@pytest.mark.parametrize("format", ["fits", "ascii.ecsv", "ascii.csv"])
def test_lightcurve_read_write(tmp_path, lc, format):
    lc.write(tmp_path / "tmp", format=format)
    lc = LightCurve.read(tmp_path / "tmp", format=format)

    # Check if time-related info round-trips
    time = lc.time
    assert time.scale == "utc"
    assert time.format == "mjd"
    assert_allclose(time.mjd, [55198, 55202.5])


@requires_dependency("matplotlib")
def test_lightcurve_plot(lc):
    with mpl_plot_check():
        lc.plot()


@pytest.mark.parametrize("flux_unit", ["cm-2 s-1"])
def test_lightcurve_plot_flux(lc, flux_unit):
    f, ferr = lc._get_fluxes_and_errors(flux_unit)
    assert_allclose(f, [1e-11, 3e-11])
    assert_allclose(ferr, ([0.1e-11, 0.3e-11], [0.1e-11, 0.3e-11]))


@pytest.mark.parametrize("flux_unit", ["cm-2 s-1"])
def test_lightcurve_plot_flux_ul(lc, flux_unit):
    is_ul, ful = lc._get_flux_uls(flux_unit)
    assert_allclose(is_ul, [False, True])
    assert_allclose(ful, [np.nan, 3.6e-11])


def test_lightcurve_plot_time(lc):
    t, terr = lc._get_times_and_errors("mjd")
    assert np.array_equal(t, [55198.0, 55202.5])
    assert np.array_equal(terr, [[1.0, 3.5], [1.0, 3.5]])

    t, terr = lc._get_times_and_errors("iso")
    assert np.array_equal(
        t, [datetime.datetime(2010, 1, 2), datetime.datetime(2010, 1, 6, 12)]
    )
    assert np.array_equal(
        terr,
        [
            [datetime.timedelta(1), datetime.timedelta(3.5)],
            [datetime.timedelta(1), datetime.timedelta(3.5)],
        ],
    )



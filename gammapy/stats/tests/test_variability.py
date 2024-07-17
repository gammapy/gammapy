# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Column, Table
from astropy.time import Time
from gammapy.estimators import FluxPoints
from gammapy.stats.variability import (
    TimmerKonig_lightcurve_simulator,
    compute_chisq,
    compute_flux_doubling,
    compute_fpp,
    compute_fvar,
    structure_function,
)
from gammapy.utils.testing import assert_quantity_allclose


@pytest.fixture(scope="session")
def lc_table():
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

    return table


def lc():
    meta = dict(TIMESYS="utc", SED_TYPE="flux")

    table = Table(
        meta=meta,
        data=[
            Column(Time(["2010-01-01", "2010-01-03", "2010-01-07"]).mjd, "time_min"),
            Column(Time(["2010-01-03", "2010-01-07", "2010-01-10"]).mjd, "time_max"),
            Column([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], "e_min", unit="TeV"),
            Column([[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]], "e_max", unit="TeV"),
            Column(
                [[1e-11, 4e-12], [3e-11, np.nan], [1e-11, 1e-12]],
                "flux",
                unit="cm-2 s-1",
            ),
            Column(
                [[0.1e-11, 0.4e-12], [0.3e-11, np.nan], [0.1e-11, 0.1e-12]],
                "flux_err",
                unit="cm-2 s-1",
            ),
            Column(
                [[np.nan, np.nan], [3.6e-11, 1e-11], [1e-11, 1e-12]],
                "flux_ul",
                unit="cm-2 s-1",
            ),
            Column([[False, False], [True, True], [True, True]], "is_ul"),
            Column([[True, True], [True, True], [True, True]], "success"),
        ],
    )

    return FluxPoints.from_table(table=table, format="lightcurve")


def test_lightcurve_fvar():
    flux = np.array([[1e-11, 4e-12], [3e-11, np.nan], [1e-11, 1e-12]])
    flux_err = np.array([[0.1e-11, 0.4e-12], [0.3e-11, np.nan], [0.1e-11, 0.1e-12]])

    time_id = 0

    fvar, fvar_err = compute_fvar(flux, flux_err, axis=time_id)

    assert_allclose(fvar, [0.68322763, 0.84047606])
    assert_allclose(fvar_err, [0.06679978, 0.08285806])


def test_lightcurve_fpp():
    flux = np.array([[1e-11, 4e-12], [3e-11, np.nan], [1e-11, 1e-12]])
    flux_err = np.array([[0.1e-11, 0.4e-12], [0.3e-11, np.nan], [0.1e-11, 0.1e-12]])

    fpp, fpp_err = compute_fpp(flux, flux_err)

    assert_allclose(fpp, [1.19448734, 0.11661904])
    assert_allclose(fpp_err, [0.06648574, 0.10099505])

    flux_1d = np.array([1e-11, 3e-11, 1e-11])
    flux_err_1d = np.array([0.1e-11, 0.3e-11, 0.1e-11])

    fpp_1d, fpp_err_1d = compute_fpp(flux_1d, flux_err_1d)

    assert_allclose(fpp_1d, [1.19448734])
    assert_allclose(fpp_err_1d, [0.06648574])


def test_lightcurve_chisq(lc_table):
    flux = lc_table["flux"].astype("float64")
    chi2, pval = compute_chisq(flux)
    assert_quantity_allclose(chi2, 1e-11)
    assert_quantity_allclose(pval, 0.999997476867478)


def test_lightcurve_flux_doubling():
    flux = np.array(
        [
            [1e-11, 4e-12],
            [3e-11, np.nan],
            [1e-11, 1e-12],
            [0.8e-11, 0.8e-12],
            [1e-11, 1e-12],
        ]
    )
    flux_err = np.array(
        [
            [0.1e-11, 0.4e-12],
            [0.3e-11, np.nan],
            [0.1e-11, 0.1e-12],
            [0.08e-11, 0.8e-12],
            [0.1e-11, 0.1e-12],
        ]
    )
    time = (
        np.array(
            [6.31157019e08, 6.31160619e08, 6.31164219e08, 6.31171419e08, 6.31178419e08]
        )
        * u.s
    )
    time_id = 0

    dtime_dict = compute_flux_doubling(flux, flux_err, time, axis=time_id)

    dtime = dtime_dict["doubling"]
    dtime_err = dtime_dict["doubling_err"]
    assert_allclose(
        dtime,
        [2271.34711286, 21743.98603654] * u.s,
    )
    assert_allclose(dtime_err, [425.92375713, 242.80234065] * u.s)


def test_tk_function():
    time_series, time_axis = TimmerKonig_lightcurve_simulator(
        lambda x: x ** (-3), 20, 1 * u.s
    )

    def temp(x, norm, index):
        return norm * x ** (-index)

    params = {"norm": 1.5, "index": 3}

    time_series2, time_axis2 = TimmerKonig_lightcurve_simulator(
        temp, 15, 1 * u.h, power_spectrum_params=params
    )

    assert len(time_series) == 20
    assert isinstance(time_axis, u.Quantity)
    assert time_axis.unit == u.s
    assert len(time_series2) == 15


def test_tk_nchunks():
    time_series, time_axis = TimmerKonig_lightcurve_simulator(
        lambda x: x ** (-1.5), 21, 1 * u.s, nchunks=100
    )
    time_series2, time_axis2 = TimmerKonig_lightcurve_simulator(
        lambda x: x ** (-1.5), 21, 1 * u.s, nchunks=20
    )

    with pytest.raises(TypeError):
        TimmerKonig_lightcurve_simulator(lambda x: x ** (-3), 20, 1 * u.s, nchunks=0.5)

    with pytest.raises(TypeError):
        TimmerKonig_lightcurve_simulator(lambda x: x ** (-3), 20.5, 1 * u.s)

    assert len(time_series) == len(time_series2) == 21


def test_tk_mean():
    time_series, time_axis = TimmerKonig_lightcurve_simulator(
        lambda x: x ** (-1.5), 2000, 1 * u.s, mean=2.5, std=0.5
    )

    time_series2, time_axis2 = TimmerKonig_lightcurve_simulator(
        lambda x: x ** (-1.5),
        2000,
        1 * u.s,
        mean=1e-7 * (u.cm**-2),
        std=5e-8 * (u.cm**-2),
    )

    time_series3, time_axis3 = TimmerKonig_lightcurve_simulator(
        lambda x: x ** (-1.5), 2000, 1 * u.s, mean=2.5, std=0.5, poisson=True
    )

    with pytest.raises(Warning):
        TimmerKonig_lightcurve_simulator(
            lambda x: x ** (-3), 20, 1 * u.s, mean=0.5, poisson=True
        )

    assert_allclose(time_series.mean(), 2.5)
    assert_allclose(time_series.std(), 0.5)
    assert_allclose(time_series2.mean(), 1e-7 * (u.cm**-2))
    assert_allclose(time_series2.std(), 5e-8 * (u.cm**-2))
    assert_allclose(time_series3.mean(), 2.5, rtol=0.1)
    assert_allclose(time_series3.std(), 1, rtol=1)


def test_structure_function():
    flux = np.array(
        [
            [1e-11, 4e-12],
            [3e-11, 2.5e-12],
            [1e-11, 1e-12],
            [0.8e-11, 0.8e-12],
            [1e-11, 1e-12],
        ]
    )
    flux_err = np.array(
        [
            [0.1e-11, 0.4e-12],
            [0.3e-11, 0.2e-12],
            [0.1e-11, 0.1e-12],
            [0.08e-11, 0.08e-12],
            [0.1e-11, 0.1e-12],
        ]
    )
    time = (
        np.array(
            [6.31157019e08, 6.31160619e08, 6.31164219e08, 6.31171419e08, 6.31178419e08]
        )
        * u.s
    )

    sf, distances = structure_function(flux, flux_err, time)

    assert_allclose(
        sf,
        [
            [4.00000000e-22, 2.25000000e-24],
            [4.00000000e-24, 4.00000000e-26],
            [2.00000000e-24, 4.52000000e-24],
            [4.84000000e-22, 2.89000000e-24],
            [0.00000000e00, 0.00000000e00],
            [4.00000000e-24, 1.02400000e-23],
            [4.00000000e-22, 2.25000000e-24],
            [0.00000000e00, 9.00000000e-24],
        ],
    )

    assert_allclose(
        distances,
        [3600.0, 7000.0, 7200.0, 10800.0, 14200.0, 14400.0, 17800.0, 21400.0] * u.s,
    )

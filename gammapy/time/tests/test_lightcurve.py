# Licensed under a 3-clause BSD style license - see LICENSE.rst
from datetime import datetime, timedelta
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.time import Time
import astropy.units as u
from astropy.table import Table, Column
from ...utils.testing import requires_data, requires_dependency, mpl_plot_check
from ...utils.testing import assert_quantity_allclose
from ...spectrum.tests.test_flux_point_estimator import (
    simulate_spectrum_dataset,
    simulate_map_dataset,
)
from ...spectrum.models import PowerLaw
from ..lightcurve import LightCurve
from ..lightcurve_estimator import LightCurveEstimator


# time time_min time_max flux flux_err flux_ul
# 48705.1757 48705.134 48705.2174 0.57 0.29 nan
# 48732.89195 48732.8503 48732.9336 0.39 0.29 nan
# 48734.0997 48734.058 48734.1414 0.48 0.29 nan
# 48738.98535 48738.9437 48739.027 nan nan 0.97
# 48741.0259 48740.9842 48741.0676 0.34 0.29 nan


@pytest.fixture(scope="session")
def lc():
    # table = Table()
    # time_ref = Time('2010-01-01')
    # meta = time_ref_to_dict(time_ref)

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
def test_lightcurve_read_write(tmpdir, lc, format):
    filename = str(tmpdir / "spam")

    lc.write(filename, format=format)
    lc = LightCurve.read(filename, format=format)

    # Check if time-related info round-trips
    time = lc.time
    assert time.scale == "utc"
    assert time.format == "mjd"
    assert_allclose(time.mjd, [55198, 55202.5])


def test_lightcurve_fvar(lc):
    fvar, fvar_err = lc.compute_fvar()
    assert_allclose(fvar, 0.6982120021884471)
    # Note: the following tolerance is very low in the next assert,
    # because results differ by ~ 1e-3 between different machines
    assert_allclose(fvar_err, 0.07905694150420949, rtol=1e-2)


def test_lightcurve_chisq(lc):
    chi2, pval = lc.compute_chisq()
    assert_quantity_allclose(chi2, 1.0000000000000001e-11)
    assert_quantity_allclose(pval, 0.999997476867478)


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


@requires_dependency("matplotlib")
@pytest.mark.parametrize(
    "time_format, output",
    [
        ("mjd", ([55198.0, 55202.5], ([1.0, 3.5], [1.0, 3.5]))),
        (
            "iso",
            (
                [datetime(2010, 1, 2), datetime(2010, 1, 6, 12)],
                ([timedelta(1), timedelta(3.5)], [timedelta(1), timedelta(3.5)]),
            ),
        ),
    ],
)
def test_lightcurve_plot_time(lc, time_format, output):
    t, terr = lc._get_times_and_errors(time_format)
    assert np.array_equal(t, output[0])
    assert np.array_equal(terr, output[1])


def get_spectrum_datasets():
    model = PowerLaw()
    dataset_1 = simulate_spectrum_dataset(model=model, random_state=0)
    dataset_1.counts.meta = {
        "t_start": Time("2010-01-01T00:00:00"),
        "t_stop": Time("2010-01-01T01:00:00"),
    }

    dataset_2 = simulate_spectrum_dataset(model, random_state=1)
    dataset_2.counts.meta = {
        "t_start": Time("2010-01-01T01:00:00"),
        "t_stop": Time("2010-01-01T02:00:00"),
    }

    return [dataset_1, dataset_2]


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets():
    datasets = get_spectrum_datasets()

    estimator = LightCurveEstimator(datasets, norm_n_values=3)
    lightcurve = estimator.run(e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV)

    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["e_ref"], [10, 10])
    assert_allclose(lightcurve.table["e_min"], [1, 1])
    assert_allclose(lightcurve.table["e_max"], [100, 100])
    assert_allclose(lightcurve.table["ref_dnde"], [1e-14, 1e-14])
    assert_allclose(lightcurve.table["ref_flux"], [9.9e-13, 9.9e-13])
    assert_allclose(lightcurve.table["ref_eflux"], [4.60517e-12, 4.60517e-12])
    assert_allclose(lightcurve.table["ref_e2dnde"], [1e-12, 1e-12])
    assert_allclose(lightcurve.table["loglike"], [23.302288, 22.457766], rtol=1e-5)
    assert_allclose(lightcurve.table["norm"], [0.988127, 0.948108], rtol=1e-5)
    assert_allclose(lightcurve.table["norm_err"], [0.043985, 0.043498], rtol=1e-4)
    assert_allclose(lightcurve.table["counts"], [2281, 2222])
    assert_allclose(lightcurve.table["norm_errp"], [0.044231, 0.04377], rtol=1e-5)
    assert_allclose(lightcurve.table["norm_errn"], [0.04374, 0.043226], rtol=1e-5)
    assert_allclose(lightcurve.table["norm_ul"], [1.077213, 1.036237], rtol=1e-5)
    assert_allclose(lightcurve.table["sqrt_ts"], [37.16802, 37.168033], rtol=1e-5)
    assert_allclose(lightcurve.table["ts"], [1381.461738, 1381.462675], rtol=1e-5)
    assert_allclose(lightcurve.table[0]["norm_scan"], [0.2, 1.0, 5.0])
    assert_allclose(
        lightcurve.table[0]["dloglike_scan"],
        [444.426957, 23.375417, 3945.382802],
        rtol=1e-5,
    )


def get_map_datasets():
    dataset_1 = simulate_map_dataset(random_state=0)
    dataset_1.counts.meta = {
        "t_start": Time("2010-01-01T00:00:00"),
        "t_stop": Time("2010-01-01T01:00:00"),
    }

    dataset_2 = simulate_map_dataset(random_state=1)
    dataset_2.counts.meta = {
        "t_start": Time("2010-01-01T01:00:00"),
        "t_stop": Time("2010-01-01T02:00:00"),
    }

    return [dataset_1, dataset_2]


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_map_datasets():
    datasets = get_map_datasets()

    estimator = LightCurveEstimator(datasets, source="source")
    steps = ["err", "counts", "ts", "norm-scan"]
    lightcurve = estimator.run(
        e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps
    )

    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["e_ref"], [10, 10])
    assert_allclose(lightcurve.table["e_min"], [1, 1])
    assert_allclose(lightcurve.table["e_max"], [100, 100])
    assert_allclose(lightcurve.table["ref_dnde"], [1e-13, 1e-13])
    assert_allclose(lightcurve.table["ref_flux"], [9.9e-12, 9.9e-12])
    assert_allclose(lightcurve.table["ref_eflux"], [4.60517e-11, 4.60517e-11])
    assert_allclose(lightcurve.table["ref_e2dnde"], [1e-11, 1e-11])
    assert_allclose(lightcurve.table["loglike"], [-86541.447142, -89740.436161])
    assert_allclose(lightcurve.table["norm_err"], [0.042729, 0.042469], rtol=1e-4)
    assert_allclose(lightcurve.table["counts"], [46648, 47321])
    assert_allclose(lightcurve.table["sqrt_ts"], [54.034112, 53.920883], rtol=1e-5)
    assert_allclose(lightcurve.table["ts"], [2919.685309, 2907.461611], rtol=1e-5)

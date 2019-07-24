# Licensed under a 3-clause BSD style license - see LICENSE.rst
from datetime import datetime, timedelta
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
import astropy.units as u
from astropy.table import Table, Column
from regions import CircleSkyRegion
from ...utils.testing import requires_data, requires_dependency, mpl_plot_check
from ...utils.testing import assert_quantity_allclose
from ...utils.energy import energy_logspace
from ...data import DataStore
from ...spectrum import SpectrumExtraction, SpectrumDatasetOnOff, CountsSpectrum
from ...spectrum.tests.test_flux_point_estimator import simulate_spectrum_dataset, simulate_map_dataset
from ...irf import EnergyDispersion, EffectiveAreaTable
from ...spectrum.models import PowerLaw, PowerLaw2
from ...background import ReflectedRegionsBackgroundEstimator
from ..lightcurve import LightCurve, LightCurveEstimator
from ..lightcurve_estimator import LightCurveEstimator3D
from ...cube import MapDataset
from ...cube.models import SkyModel



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


# TODO: Reuse fixtures from spectrum tests
@pytest.fixture(scope="session")
def spec_extraction():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids = [23523, 23526]
    observations = data_store.get_observations(obs_ids)

    target_position = SkyCoord(ra=83.63308, dec=22.01450, unit="deg")
    on_region_radius = Angle("0.11 deg")
    on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

    bkg_estimator = ReflectedRegionsBackgroundEstimator(
        on_region=on_region, observations=observations
    )
    bkg_estimator.run()

    e_reco = energy_logspace(0.2, 100, 51, unit="TeV")  # fine binning
    e_true = energy_logspace(0.05, 100, 201, unit="TeV")
    extraction = SpectrumExtraction(
        observations=observations,
        bkg_estimate=bkg_estimator.result,
        containment_correction=False,
        e_reco=e_reco,
        e_true=e_true,
    )
    extraction.run()
    extraction.compute_energy_threshold(method_lo="area_max", area_percent_lo=10.0)
    return extraction


@requires_data()
def test_lightcurve_estimator(spec_extraction):
    lc_estimator = LightCurveEstimator(spec_extraction)

    intervals = []
    for obs in spec_extraction.observations:
        intervals.append([obs.events.time[0], obs.events.time[-1]])

    model = PowerLaw(
        index=2.3 * u.Unit(""),
        amplitude=3.4e-11 * u.Unit("1 / (cm2 s TeV)"),
        reference=1 * u.TeV,
    )

    lc = lc_estimator.light_curve(
        time_intervals=intervals, spectral_model=model, energy_range=[0.5, 100] * u.TeV
    )
    table = lc.table

    assert isinstance(lc.table["time_min"][0], type(intervals[0][0].value))

    assert_quantity_allclose(len(table), 2)

    assert_allclose(table["flux"][0], 4.333763e-11, rtol=5e-3)
    assert_allclose(table["flux"][-1], 3.527114e-11, rtol=5e-3)

    assert_allclose(table["flux_err"][0], 4.135581e-12, rtol=5e-3)
    assert_allclose(table["flux_err"][-1], 3.657088e-12, rtol=5e-3)

    # TODO: change dataset and also add LC point with weak signal
    # or even negative excess that is an UL
    assert_allclose(table["flux_ul"][0], 5.550045e-11, rtol=5e-3)
    assert not table["is_ul"][0]

    # same but with threshold equal to 2 TeV
    lc = lc_estimator.light_curve(
        time_intervals=intervals, spectral_model=model, energy_range=[2, 100] * u.TeV
    )
    table = lc.table

    assert_allclose(table["flux"][0], 5.0902e-12, rtol=5e-3)

    # TODO: add test exercising e_reco selection
    # TODO: add asserts on all measured quantities


@requires_data()
def test_lightcurve_interval_maker(spec_extraction):
    table = LightCurveEstimator.make_time_intervals_fixes(500, spec_extraction)
    intervals = list(zip(table["t_start"], table["t_stop"]))

    assert len(intervals) == 9
    t = intervals[0]
    assert_allclose(t[1].value - t[0].value, 500 / (24 * 3600), rtol=1e-5)


@requires_data()
def test_lightcurve_adaptative_interval_maker(spec_extraction):
    lc_estimator = LightCurveEstimator(spec_extraction)
    separator = [
        Time((53343.94050200008 + 53343.952979345195) / 2, scale="tt", format="mjd")
    ]
    table = lc_estimator.make_time_intervals_min_significance(
        significance=3,
        significance_method="lima",
        energy_range=[0.2, 100] * u.TeV,
        spectrum_extraction=spec_extraction,
        separators=separator,
    )
    assert_allclose(table["significance"] >= 3, True)
    assert_allclose(table["t_start"][5].value, 53343.927761, rtol=1e-10)
    assert_allclose(table["alpha"][5], 0.0833333, rtol=1e-5)
    assert len(table) == 52
    assert_allclose(table["t_start"][0].value, 53343.922392, rtol=1e-10)
    assert_allclose(table["t_stop"][-1].value, 53343.973528, rtol=1e-10)
    val = (table["t_start"] < separator[0]) & (table["t_stop"] > separator[0])
    assert_allclose(val, False)


def get_spectrum_datasets():
    model = PowerLaw()
    dataset_1 = simulate_spectrum_dataset(model=model, random_state=0)
    dataset_1.counts.meta = {
        "t_start": Time('2010-01-01T00:00:00'),
        "t_stop": Time('2010-01-01T01:00:00'),
    }

    dataset_2 = simulate_spectrum_dataset(model, random_state=1)
    dataset_2.counts.meta = {
        "t_start": Time('2010-01-01T01:00:00'),
        "t_stop": Time('2010-01-01T02:00:00'),
    }

    return [dataset_1, dataset_2]


def test_lightcurve_estimator_spectrum_datasets():
    datasets = get_spectrum_datasets()

    estimator = LightCurveEstimator3D(datasets, norm_n_values=3)
    lightcurve = estimator.run(e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV)

    assert_allclose(lightcurve.table["time_min"], [55197., 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["e_ref"], [10, 10])
    assert_allclose(lightcurve.table["e_min"], [1 , 1])
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
    assert_allclose(lightcurve.table[0]["norm_scan"], [0.2, 1., 5.])
    assert_allclose(lightcurve.table[0]["dloglike_scan"], [444.426957, 23.375417, 3945.382802], rtol=1e-5)


def get_map_datasets():
    dataset_1 = simulate_map_dataset(random_state=0)
    dataset_1.counts.meta = {
        "t_start": Time('2010-01-01T00:00:00'),
        "t_stop": Time('2010-01-01T01:00:00'),
    }

    dataset_2 = simulate_map_dataset(random_state=1)
    dataset_2.counts.meta = {
        "t_start": Time('2010-01-01T01:00:00'),
        "t_stop": Time('2010-01-01T02:00:00'),
    }

    return [dataset_1, dataset_2]


def test_lightcurve_estimator_map_datasets():
    datasets = get_map_datasets()

    estimator = LightCurveEstimator3D(datasets, source="source")
    steps = ["err", "counts", "ts", "norm-scan"]
    lightcurve = estimator.run(e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps)

    assert_allclose(lightcurve.table["time_min"], [55197., 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["e_ref"], [10, 10])
    assert_allclose(lightcurve.table["e_min"], [1 , 1])
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


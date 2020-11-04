import datetime
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Column, Table
from astropy.time import Time
from gammapy.data import GTI
from gammapy.datasets import Datasets
from gammapy.estimators import LightCurve, LightCurveEstimator
from gammapy.estimators.tests.test_flux_point_estimator import (
    simulate_map_dataset,
    simulate_spectrum_dataset,
)
from gammapy.maps import RegionNDMap
from gammapy.modeling.models import FoVBackgroundModel, PowerLawSpectralModel, SkyModel
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


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


@pytest.fixture(scope="session")
def lc_2d():
    meta = dict(TIMESYS="utc")

    table = Table(
        meta=meta,
        data=[
            Column(Time(["2010-01-01", "2010-01-03"]).mjd, "time_min"),
            Column(Time(["2010-01-03", "2010-01-10"]).mjd, "time_max"),
            Column([[1.0, 2.0], [1.0, 2.0]], "e_min", unit="TeV"),
            Column([[2.0, 4.0], [2.0, 4.0]], "e_max", unit="TeV"),
            Column([[1e-11, 1e-12], [3e-11, 3e-12]], "flux", unit="cm-2 s-1"),
            Column([[0.1e-11, 1e-13], [0.3e-11, 3e-13]], "flux_err", unit="cm-2 s-1"),
            Column([[np.nan, np.nan], [3.6e-11, 3.6e-12]], "flux_ul", unit="cm-2 s-1"),
            Column([[False, False], [True, True]], "is_ul"),
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
def test_lightcurve_plot(lc, lc_2d):
    with mpl_plot_check():
        lc.plot()
    with mpl_plot_check():
        lc_2d.plot(energy_index=1)


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


def get_spectrum_datasets():
    model = SkyModel(spectral_model=PowerLawSpectralModel())
    dataset_1 = simulate_spectrum_dataset(model=model, random_state=0)
    dataset_1._name = "dataset_1"
    gti1 = GTI.create("0h", "1h", "2010-01-01T00:00:00")
    dataset_1.gti = gti1

    dataset_2 = simulate_spectrum_dataset(model=model, random_state=1)
    dataset_2._name = "dataset_2"
    gti2 = GTI.create("1h", "2h", "2010-01-01T00:00:00")
    dataset_2.gti = gti2

    return [dataset_1, dataset_2]


@requires_data()
@requires_dependency("iminuit")
def test_group_datasets_in_time_interval():
    # Doing a LC on one hour bin
    datasets = Datasets(get_spectrum_datasets())
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]

    group_table = datasets.gti.group_table(time_intervals)

    assert len(group_table) == 2
    assert_allclose(group_table["time_min"], [55197.0, 55197.04166666667])
    assert_allclose(group_table["time_max"], [55197.04166666667, 55197.083333333336])
    assert_allclose(group_table["group_idx"], [0, 1])


@requires_data()
@requires_dependency("iminuit")
def test_group_datasets_in_time_interval_outflows():
    datasets = Datasets(get_spectrum_datasets())
    # Check Overflow
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T00:55:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]

    group_table = datasets.gti.group_table(time_intervals)
    assert group_table["bin_type"][0] == "overflow"

    # Check underflow
    time_intervals = [
        Time(["2010-01-01T00:05:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]
    group_table = datasets.gti.group_table(time_intervals)
    assert group_table["bin_type"][0] == "underflow"


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets():
    # Doing a LC on one hour bin
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]

    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV, norm_n_values=3, time_intervals=time_intervals
    )
    lightcurve = estimator.run(datasets)
    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["e_ref"], [[5.623413], [5.623413]])
    assert_allclose(lightcurve.table["e_min"], [[1], [1]])
    assert_allclose(lightcurve.table["e_max"], [[31.622777], [31.622777]])
    assert_allclose(
        lightcurve.table["ref_dnde"], [[3.162278e-14], [3.162278e-14]], rtol=1e-5
    )
    assert_allclose(
        lightcurve.table["ref_flux"], [[9.683772e-13], [9.683772e-13]], rtol=1e-5
    )
    assert_allclose(
        lightcurve.table["ref_eflux"], [[3.453878e-12], [3.453878e-12]], rtol=1e-5
    )
    assert_allclose(lightcurve.table["ref_e2dnde"], [[1e-12], [1e-12]], rtol=1e-5)
    assert_allclose(lightcurve.table["stat"], [[16.824042], [17.391981]], rtol=1e-5)
    assert_allclose(lightcurve.table["norm"], [[0.911963], [0.9069318]], rtol=1e-2)
    assert_allclose(lightcurve.table["norm_err"], [[0.057769], [0.057835]], rtol=1e-2)
    assert_allclose(lightcurve.table["counts"], [[791], [784]])
    assert_allclose(lightcurve.table["norm_errp"], [[0.058398], [0.058416]], rtol=1e-2)
    assert_allclose(lightcurve.table["norm_errn"], [[0.057144], [0.057259]], rtol=1e-2)
    assert_allclose(lightcurve.table["norm_ul"], [[1.029989], [1.025061]], rtol=1e-2)
    assert_allclose(lightcurve.table["sqrt_ts"], [[19.384781], [19.161769]], rtol=1e-2)
    assert_allclose(lightcurve.table["ts"], [[375.769735], [367.173374]], rtol=1e-2)
    assert_allclose(lightcurve.table[0]["norm_scan"], [[0.2, 1.0, 5.0]])
    assert_allclose(
        lightcurve.table[0]["stat_scan"],
        [[224.058304, 19.074405, 2063.75636]],
        rtol=1e-5,
    )


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets_2_energy_bins():
    # Doing a LC on one hour bin
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]

    estimator = LightCurveEstimator(
        energy_edges=[1, 5, 30] * u.TeV, norm_n_values=3, time_intervals=time_intervals
    )
    lightcurve = estimator.run(datasets)
    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(
        lightcurve.table["e_ref"], [[2.238721, 12.589254], [2.238721, 12.589254]]
    )
    assert_allclose(lightcurve.table["e_min"], [[1, 5.011872], [1, 5.011872]])
    assert_allclose(
        lightcurve.table["e_max"], [[5.011872, 31.622777], [5.011872, 31.622777]]
    )
    assert_allclose(
        lightcurve.table["ref_dnde"],
        [[1.995262e-13, 6.309573e-15], [1.995262e-13, 6.309573e-15]],
        rtol=1e-5,
    )
    assert_allclose(
        lightcurve.table["stat"],
        [[8.234951, 8.30321], [2.037205, 15.300507]],
        rtol=1e-5,
    )
    assert_allclose(
        lightcurve.table["norm"],
        [[0.894723, 0.967419], [0.914283, 0.882351]],
        rtol=1e-2,
    )
    assert_allclose(
        lightcurve.table["norm_err"],
        [[0.065905, 0.121288], [0.06601, 0.119457]],
        rtol=1e-2,
    )
    assert_allclose(lightcurve.table["counts"], [[669.0, 122.0], [667.0, 117.0]])
    assert_allclose(
        lightcurve.table["norm_errp"],
        [[0.06664, 0.124741], [0.066815, 0.122832]],
        rtol=1e-2,
    )
    assert_allclose(
        lightcurve.table["norm_errn"],
        [[0.065176, 0.117904], [0.065212, 0.116169]],
        rtol=1e-2,
    )
    assert_allclose(
        lightcurve.table["norm_ul"],
        [[1.029476, 1.224117], [1.049283, 1.134874]],
        rtol=1e-2,
    )
    assert_allclose(
        lightcurve.table["sqrt_ts"],
        [[16.233236, 10.608376], [16.609784, 9.557339]],
        rtol=1e-2,
    )
    assert_allclose(
        lightcurve.table[0]["norm_scan"], [[0.2, 1.0, 5.0], [0.2, 1.0, 5.0]]
    )
    assert_allclose(
        lightcurve.table[0]["stat_scan"],
        [[153.880281, 10.701492, 1649.609684], [70.178023, 8.372913, 414.146676]],
        rtol=1e-5,
    )


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets_withmaskfit():
    # Doing a LC on one hour bin
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]

    energy_min_fit = 1 * u.TeV
    energy_max_fit = 3 * u.TeV
    for dataset in datasets:
        geom = dataset.counts.geom
        data = geom.energy_mask(energy_min=energy_min_fit, energy_max=energy_max_fit)
        dataset.mask_fit = RegionNDMap.from_geom(geom, data=data, dtype=bool)

    selection = ["scan"]
    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV,
        norm_n_values=3,
        time_intervals=time_intervals,
        selection_optional=selection,
    )
    lightcurve = estimator.run(datasets)
    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["stat"], [[6.603043], [0.421051]], rtol=1e-3)
    assert_allclose(lightcurve.table["norm"], [[0.885124], [0.967054]], rtol=1e-3)


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets_default():
    # Test default time interval: each time interval is equal to the gti of each dataset, here one hour
    datasets = get_spectrum_datasets()
    selection = ["scan"]
    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV, norm_n_values=3, selection_optional=selection
    )
    lightcurve = estimator.run(datasets)
    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["norm"], [[0.911963], [0.906931]], rtol=1e-3)


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets_notordered():
    # Test that if the time intervals given are not ordered in time, it is first ordered correctly and then
    # compute as expected
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
    ]
    estimator = LightCurveEstimator(
        energy_edges=[1, 100] * u.TeV,
        norm_n_values=3,
        time_intervals=time_intervals,
        selection_optional=["scan"],
    )
    lightcurve = estimator.run(datasets)
    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["norm"], [[0.911963], [0.906931]], rtol=1e-3)


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets_largerbin():
    # Test all dataset in a single LC bin, here two hours
    datasets = get_spectrum_datasets()
    time_intervals = [Time(["2010-01-01T00:00:00", "2010-01-01T02:00:00"])]
    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV,
        norm_n_values=3,
        time_intervals=time_intervals,
        selection_optional=["scan"],
    )
    lightcurve = estimator.run(datasets)

    assert_allclose(lightcurve.table["time_min"], [55197.0])
    assert_allclose(lightcurve.table["time_max"], [55197.083333])
    assert_allclose(lightcurve.table["e_ref"][0], [5.623413])
    assert_allclose(lightcurve.table["e_min"][0], [1])
    assert_allclose(lightcurve.table["e_max"][0], [31.622777])
    assert_allclose(lightcurve.table["ref_dnde"][0], [3.162278e-14], rtol=1e-5)
    assert_allclose(lightcurve.table["ref_flux"][0], [9.683772e-13], rtol=1e-5)
    assert_allclose(lightcurve.table["ref_eflux"][0], [3.453878e-12], rtol=1e-5)
    assert_allclose(lightcurve.table["ref_e2dnde"][0], [1e-12], rtol=1e-5)
    assert_allclose(lightcurve.table["stat"][0], [34.219808], rtol=1e-5)
    assert_allclose(lightcurve.table["norm"][0], [0.909646], rtol=1e-5)
    assert_allclose(lightcurve.table["norm_err"][0], [0.040874], rtol=1e-3)
    assert_allclose(lightcurve.table["ts"][0], [742.939324], rtol=1e-4)


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets_timeoverlaped():
    # Check that it returns a ValueError if the time intervals overlapped
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:30:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]
    with pytest.raises(ValueError) as excinfo:
        estimator = LightCurveEstimator(norm_n_values=3, time_intervals=time_intervals)
        estimator.run(datasets)

    msg = "Overlapping time bins"
    assert str(excinfo.value) == msg


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets_gti_not_include_in_time_intervals():
    # Check that it returns a ValueError if the time intervals are smaller than the dataset GTI.
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T00:05:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T01:05:00"]),
    ]
    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV,
        norm_n_values=3,
        time_intervals=time_intervals,
        selection_optional=["scan"],
    )
    with pytest.raises(ValueError) as excinfo:
        estimator.run(datasets)
    msg = "LightCurveEstimator: No datasets in time intervals"
    assert str(excinfo.value) == msg


def get_map_datasets():
    dataset_1 = simulate_map_dataset(random_state=0, name="dataset_1")
    gti1 = GTI.create("0 h", "1 h", "2010-01-01T00:00:00")
    dataset_1.gti = gti1

    dataset_2 = simulate_map_dataset(random_state=1, name="dataset_2")
    gti2 = GTI.create("1 h", "2 h", "2010-01-01T00:00:00")
    dataset_2.gti = gti2

    model = dataset_1.models["source"].copy("test_source")
    bkg_model_1 = FoVBackgroundModel(dataset_name="dataset_1")
    bkg_model_2 = FoVBackgroundModel(dataset_name="dataset_2")

    dataset_1.models = [model, bkg_model_1]
    dataset_2.models = [model, bkg_model_2]

    return [dataset_1, dataset_2]


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_map_datasets():
    datasets = get_map_datasets()

    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]
    estimator = LightCurveEstimator(
        energy_edges=[1, 100] * u.TeV,
        source="test_source",
        time_intervals=time_intervals,
        selection_optional=["scan"],
    )
    lightcurve = estimator.run(datasets)
    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["e_ref"], [[10.857111], [10.857111]])
    assert_allclose(lightcurve.table["e_min"], [[1.178769], [1.178769]], rtol=1e-5)
    assert_allclose(lightcurve.table["e_max"], [[100], [100]])
    assert_allclose(
        lightcurve.table["ref_dnde"], [[8.483429e-14], [8.483429e-14]], rtol=1e-5
    )
    assert_allclose(
        lightcurve.table["ref_flux"], [[8.383429e-12], [8.383429e-12]], rtol=1e-5
    )
    assert_allclose(
        lightcurve.table["ref_eflux"], [[4.4407e-11], [4.4407e-11]], rtol=1e-5
    )
    assert_allclose(lightcurve.table["ref_e2dnde"], [[1e-11], [1e-11]], rtol=1e-5)
    assert_allclose(lightcurve.table["stat"], [[9402.778975], [9517.750207]], rtol=1e-2)
    assert_allclose(lightcurve.table["norm"], [[0.971592], [0.963286]], rtol=1e-2)
    assert_allclose(lightcurve.table["norm_err"], [[0.044643], [0.044475]], rtol=1e-2)
    assert_allclose(lightcurve.table["sqrt_ts"], [[35.880361], [35.636547]], rtol=1e-2)
    assert_allclose(lightcurve.table["ts"], [[1287.4003], [1269.963491]], rtol=1e-2)

    datasets = get_map_datasets()

    time_intervals2 = [Time(["2010-01-01T00:00:00", "2010-01-01T02:00:00"])]
    estimator2 = LightCurveEstimator(
        energy_edges=[1, 100] * u.TeV,
        source="test_source",
        time_intervals=time_intervals2,
        selection_optional=["scan"],
    )
    lightcurve2 = estimator2.run(datasets)

    assert_allclose(lightcurve2.table["time_min"][0], [55197.0])
    assert_allclose(lightcurve2.table["time_max"][0], [55197.083333])
    assert_allclose(lightcurve2.table["e_ref"][0], [10.857111], rtol=1e-5)
    assert_allclose(lightcurve2.table["e_min"][0], [1.178769], rtol=1e-5)
    assert_allclose(lightcurve2.table["e_max"][0], [100])
    assert_allclose(lightcurve2.table["ref_dnde"][0], [8.483429e-14], rtol=1e-5)
    assert_allclose(lightcurve2.table["ref_flux"][0], [8.383429e-12], rtol=1e-5)
    assert_allclose(lightcurve2.table["ref_eflux"][0], [4.4407e-11], rtol=1e-5)
    assert_allclose(lightcurve2.table["ref_e2dnde"][0], [1e-11], rtol=1e-5)
    assert_allclose(lightcurve2.table["stat"][0], [18920.54651], rtol=1e-2)
    assert_allclose(lightcurve2.table["norm"][0], [0.967438], rtol=1e-2)
    assert_allclose(lightcurve2.table["norm_err"][0], [0.031508], rtol=1e-2)
    assert_allclose(lightcurve.table["counts"][0], [2205])
    assert_allclose(lightcurve2.table["ts"][0], [2557.346464], rtol=1e-2)

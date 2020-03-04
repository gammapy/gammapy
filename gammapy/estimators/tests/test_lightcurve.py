import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.time import Time
from gammapy.data import GTI
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.estimators.tests.test_flux_point_estimator import (
    simulate_map_dataset,
    simulate_spectrum_dataset,
)
from gammapy.estimators import LightCurveEstimator
from gammapy.utils.testing import requires_data, requires_dependency


def get_spectrum_datasets():
    model = SkyModel(spectral_model=PowerLawSpectralModel())
    dataset_1 = simulate_spectrum_dataset(model=model, random_state=0)
    dataset_1._name = "dataset_1"
    gti1 = GTI.create("0h", "1h", "2010-01-01T00:00:00")
    dataset_1.gti = gti1

    dataset_2 = simulate_spectrum_dataset(model, random_state=1)
    dataset_2._name = "dataset_2"
    gti2 = GTI.create("1h", "2h", "2010-01-01T00:00:00")
    dataset_2.gti = gti2

    return [dataset_1, dataset_2]


@requires_data()
@requires_dependency("iminuit")
def test_group_datasets_in_time_interval():
    # Doing a LC on one hour bin
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]
    estimator = LightCurveEstimator(
        datasets, norm_n_values=3, time_intervals=time_intervals
    )
    steps = ["err", "counts", "ts", "norm-scan"]
    estimator.run(e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps)

    assert len(estimator.group_table_info) == 2
    assert estimator.group_table_info["Name"][0] == "dataset_1"
    assert_allclose(estimator.group_table_info["Tstart"], [55197.0, 55197.04166666667])
    assert_allclose(
        estimator.group_table_info["Tstop"], [55197.04166666667, 55197.083333333336]
    )
    assert_allclose(estimator.group_table_info["Group_ID"], [0, 1])


@requires_data()
@requires_dependency("iminuit")
def test_group_datasets_in_time_interval_outflows():
    datasets = get_spectrum_datasets()
    # Check Overflow
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T00:55:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]
    estimator = LightCurveEstimator(
        datasets, norm_n_values=3, time_intervals=time_intervals
    )
    steps = ["err", "counts", "ts", "norm-scan"]
    estimator.run(e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps)
    assert estimator.group_table_info["Bin_type"][0] == "Overflow"

    # Check underflow
    time_intervals = [
        Time(["2010-01-01T00:05:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]
    estimator = LightCurveEstimator(
        datasets, norm_n_values=3, time_intervals=time_intervals
    )
    steps = ["err", "counts", "ts", "norm-scan"]
    estimator.run(e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps)
    assert estimator.group_table_info["Bin_type"][0] == "Underflow"


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
        datasets, norm_n_values=3, time_intervals=time_intervals
    )
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
    assert_allclose(lightcurve.table["stat"], [23.302288, 22.457766], rtol=1e-5)
    assert_allclose(lightcurve.table["norm"], [0.988127, 0.948108], rtol=1e-5)
    assert_allclose(lightcurve.table["norm_err"], [0.043985, 0.043498], rtol=1e-4)
    assert_allclose(lightcurve.table["counts"], [2281, 2222])
    assert_allclose(lightcurve.table["norm_errp"], [0.044231, 0.04377], rtol=1e-5)
    assert_allclose(lightcurve.table["norm_errn"], [0.04374, 0.043226], rtol=1e-5)
    assert_allclose(lightcurve.table["norm_ul"], [1.077213, 1.036237], rtol=1e-5)
    assert_allclose(lightcurve.table["sqrt_ts"], [26.773925, 25.796426], rtol=1e-4)
    assert_allclose(lightcurve.table["ts"], [716.843084, 665.455601], rtol=1e-4)
    assert_allclose(lightcurve.table[0]["norm_scan"], [0.2, 1.0, 5.0])
    assert_allclose(
        lightcurve.table[0]["stat_scan"],
        [444.426957, 23.375417, 3945.382802],
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

    e_min_fit = 1 * u.TeV
    e_max_fit = 3 * u.TeV
    for dataset in datasets:
        mask_fit = dataset.counts.energy_mask(emin=e_min_fit, emax=e_max_fit)
        dataset.mask_fit = mask_fit

    steps = ["err", "counts", "ts", "norm-scan"]
    estimator = LightCurveEstimator(
        datasets, norm_n_values=3, time_intervals=time_intervals
    )
    lightcurve = estimator.run(
        e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps
    )
    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["stat"], [6.60304, 0.421047], rtol=1e-5)
    assert_allclose(lightcurve.table["norm"], [0.885082, 0.967022], rtol=1e-5)


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets_default():
    # Test default time interval: each time interval is equal to the gti of each dataset, here one hour
    datasets = get_spectrum_datasets()
    estimator = LightCurveEstimator(datasets, norm_n_values=3)
    steps = ["err", "counts", "ts", "norm-scan"]
    lightcurve = estimator.run(
        e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps
    )
    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["norm"], [0.988127, 0.948108], rtol=1e-5)


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
        datasets, norm_n_values=3, time_intervals=time_intervals
    )
    steps = ["err", "counts", "ts", "norm-scan"]
    lightcurve = estimator.run(
        e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps
    )
    assert_allclose(lightcurve.table["time_min"], [55197.0, 55197.041667])
    assert_allclose(lightcurve.table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(lightcurve.table["norm"], [0.988127, 0.948108], rtol=1e-5)


@requires_data()
@requires_dependency("iminuit")
def test_lightcurve_estimator_spectrum_datasets_largerbin():
    # Test all dataset in a single LC bin, here two hours
    datasets = get_spectrum_datasets()
    time_intervals = [Time(["2010-01-01T00:00:00", "2010-01-01T02:00:00"])]
    estimator = LightCurveEstimator(
        datasets, norm_n_values=3, time_intervals=time_intervals
    )
    steps = ["err", "counts", "ts", "norm-scan"]
    lightcurve = estimator.run(
        e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps
    )

    assert_allclose(lightcurve.table["time_min"], [55197.0])
    assert_allclose(lightcurve.table["time_max"], [55197.083333])
    assert_allclose(lightcurve.table["e_ref"], [10])
    assert_allclose(lightcurve.table["e_min"], [1])
    assert_allclose(lightcurve.table["e_max"], [100])
    assert_allclose(lightcurve.table["ref_dnde"], [1e-14])
    assert_allclose(lightcurve.table["ref_flux"], [9.9e-13])
    assert_allclose(lightcurve.table["ref_eflux"], [4.60517e-12])
    assert_allclose(lightcurve.table["ref_e2dnde"], [1e-12])
    assert_allclose(lightcurve.table["stat"], [46.177981], rtol=1e-5)
    assert_allclose(lightcurve.table["norm"], [0.968049], rtol=1e-5)
    assert_allclose(lightcurve.table["norm_err"], [0.030929], rtol=1e-4)
    assert_allclose(lightcurve.table["ts"], [1381.880757], rtol=1e-4)


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
        LightCurveEstimator(datasets, norm_n_values=3, time_intervals=time_intervals)
    msg = "LightCurveEstimator requires non-overlapping time bins."
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
        datasets, norm_n_values=3, time_intervals=time_intervals
    )
    with pytest.raises(ValueError) as excinfo:
        steps = ["err", "counts", "ts", "norm-scan"]
        estimator.run(e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV, steps=steps)
    msg = "LightCurveEstimator: No datasets in time intervals"
    assert str(excinfo.value) == msg


def get_map_datasets():
    dataset_1 = simulate_map_dataset(random_state=0, name="dataset_1")
    gti1 = GTI.create("0 h", "1 h", "2010-01-01T00:00:00")
    dataset_1.gti = gti1

    dataset_2 = simulate_map_dataset(random_state=1, name="dataset_2")
    gti2 = GTI.create("1 h", "2 h", "2010-01-01T00:00:00")
    dataset_2.gti = gti2

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
        datasets, source="source", time_intervals=time_intervals
    )
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
    assert_allclose(lightcurve.table["stat"], [-87412.393367, -89856.129206], rtol=1e-5)
    assert_allclose(lightcurve.table["norm_err"], [0.042259, 0.043614], rtol=1e-3)
    assert_allclose(lightcurve.table["sqrt_ts"], [38.527512, 39.489968], rtol=1e-4)
    assert_allclose(lightcurve.table["ts"], [1484.369159, 1559.457547], rtol=1e-4)

    datasets = get_map_datasets()
    time_intervals2 = [Time(["2010-01-01T00:00:00", "2010-01-01T02:00:00"])]
    estimator2 = LightCurveEstimator(
        datasets, source="source", time_intervals=time_intervals2
    )
    lightcurve2 = estimator2.run(e_ref=10 * u.TeV, e_min=1 * u.TeV, e_max=100 * u.TeV)

    assert_allclose(lightcurve2.table["time_min"], [55197.0])
    assert_allclose(lightcurve2.table["time_max"], [55197.083333])
    assert_allclose(lightcurve2.table["e_ref"], [10])
    assert_allclose(lightcurve2.table["e_min"], [1])
    assert_allclose(lightcurve2.table["e_max"], [100])
    assert_allclose(lightcurve2.table["ref_dnde"], [1e-13])
    assert_allclose(lightcurve2.table["ref_flux"], [9.9e-12])
    assert_allclose(lightcurve2.table["ref_eflux"], [4.60517e-11])
    assert_allclose(lightcurve2.table["ref_e2dnde"], [1e-11])
    assert_allclose(lightcurve2.table["stat"], [-177267.775615], rtol=1e-5)
    assert_allclose(lightcurve2.table["norm_err"], [0.030358], rtol=1e-3)
    assert_allclose(lightcurve.table["counts"], [46794, 47388])
    assert_allclose(lightcurve2.table["ts"], [3042.893291], rtol=1e-4)

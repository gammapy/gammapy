import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Column, Table
from astropy.time import Time
from astropy.timeseries import BinnedTimeSeries, BoxLeastSquares
from gammapy.data import GTI
from gammapy.datasets import Datasets
from gammapy.estimators import FluxPoints, LightCurveEstimator
from gammapy.estimators.points.tests.test_sed import (
    simulate_map_dataset,
    simulate_spectrum_dataset,
)
from gammapy.modeling import Fit
from gammapy.modeling.models import FoVBackgroundModel, PowerLawSpectralModel, SkyModel
from gammapy.utils.testing import mpl_plot_check, requires_data


@pytest.fixture(scope="session")
def lc():
    meta = dict(TIMESYS="utc", SED_TYPE="flux")

    table = Table(
        meta=meta,
        data=[
            Column(Time(["2010-01-01", "2010-01-03"]).mjd, "time_min"),
            Column(Time(["2010-01-03", "2010-01-10"]).mjd, "time_max"),
            Column([[1.0], [1.0]], "e_min", unit="TeV"),
            Column([[2.0], [2.0]], "e_max", unit="TeV"),
            Column([1e-11, 3e-11], "flux", unit="cm-2 s-1"),
            Column([0.1e-11, 0.3e-11], "flux_err", unit="cm-2 s-1"),
            Column([np.nan, 3.6e-11], "flux_ul", unit="cm-2 s-1"),
            Column([False, True], "is_ul"),
            Column([True, True], "success"),
        ],
    )

    return FluxPoints.from_table(table=table, format="lightcurve")


@pytest.fixture(scope="session")
def lc_2d():
    meta = dict(TIMESYS="utc", SED_TYPE="flux")

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

    return FluxPoints.from_table(table=table, format="lightcurve")


def test_lightcurve_str(lc):
    info_str = str(lc)
    assert "time" in info_str


def test_lightcurve_properties_time(lc):
    axis = lc.geom.axes["time"]

    assert axis.reference_time.scale == "utc"
    assert axis.reference_time.format == "mjd"

    # Time-related attributes
    time = axis.time_mid
    assert time.scale == "utc"
    assert time.format == "mjd"
    assert_allclose(time.mjd, [55198, 55202.5])

    assert_allclose(axis.time_min.mjd, [55197, 55199])
    assert_allclose(axis.time_max.mjd, [55199, 55206])

    # Note: I'm not sure why the time delta has this scale and format
    time_delta = axis.time_delta
    assert time_delta.scale == "tai"
    assert time_delta.format == "jd"
    assert_allclose(time_delta.jd, [2, 7])


def test_lightcurve_properties_flux(lc):
    table = lc.to_table(sed_type="flux", format="lightcurve")
    flux = table["flux"].quantity
    assert flux.unit == "cm-2 s-1"
    assert_allclose(flux.value, [[1e-11], [3e-11]])


# TODO: extend these tests to cover other time scales.
# In those cases, CSV should not round-trip because there
# is no header info in CSV to store the time scale!


@pytest.mark.parametrize("sed_type", ["dnde", "flux", "likelihood"])
def test_lightcurve_read_write(tmp_path, lc, sed_type):
    lc.write(tmp_path / "tmp.fits", format="lightcurve", sed_type=sed_type)

    lc = FluxPoints.read(tmp_path / "tmp.fits", format="lightcurve")

    # Check if time-related info round-trips
    axis = lc.geom.axes["time"]
    assert axis.reference_time.scale == "utc"
    assert axis.reference_time.format == "mjd"
    assert_allclose(axis.time_mid.mjd, [55198, 55202.5])


def test_lightcurve_plot(lc, lc_2d):
    with mpl_plot_check():
        lc.plot()

    with mpl_plot_check():
        lc_2d.plot(axis_name="time")


@requires_data()
def test_lightcurve_to_time_series():
    from gammapy.catalog import SourceCatalog4FGL

    catalog_4fgl = SourceCatalog4FGL("$GAMMAPY_DATA/catalogs/fermi/gll_psc_v20.fit.gz")
    lightcurve = catalog_4fgl["FGES J1553.8-5325"].lightcurve()

    table = lightcurve.to_table(sed_type="flux", format="binned-time-series")

    timeseries = BinnedTimeSeries(data=table)

    assert_allclose(timeseries.time_bin_center.mjd[0], 54863.97885336907)
    assert_allclose(timeseries.time_bin_end.mjd[0], 55045.301668796295)

    time_axis = lightcurve.geom.axes["time"]
    assert_allclose(timeseries.time_bin_end.mjd[0], time_axis.time_max.mjd[0])

    # assert that it interfaces with periodograms

    p = BoxLeastSquares.from_timeseries(
        timeseries=timeseries, signal_column_name="flux", uncertainty="flux_errp"
    )

    result = p.power(1 * u.year, 0.5 * u.year)
    assert_allclose(result.duration, 182.625 * u.d)


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
def test_lightcurve_estimator_fit_options():
    # Doing a LC on one hour bin
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]
    energy_edges = [1, 30] * u.TeV

    estimator = LightCurveEstimator(
        energy_edges=energy_edges,
        norm_n_values=3,
        time_intervals=time_intervals,
        selection_optional="all",
        fit=Fit(backend="minuit", optimize_opts=dict(tol=0.2, strategy=1)),
    )

    assert_allclose(estimator.fit.optimize_opts["tol"], 0.2)

    estimator.fit.run(datasets=datasets)
    assert_allclose(estimator.fit.minuit.tol, 0.2)


@requires_data()
def test_lightcurve_estimator_spectrum_datasets():
    # Doing a LC on one hour bin
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]

    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV,
        norm_n_values=3,
        time_intervals=time_intervals,
        selection_optional="all",
    )

    lightcurve = estimator.run(datasets)
    table = lightcurve.to_table(format="lightcurve")
    assert_allclose(table["time_min"], [55197.0, 55197.041667])
    assert_allclose(table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(table["e_ref"], [[5.623413], [5.623413]])
    assert_allclose(table["e_min"], [[1], [1]])
    assert_allclose(table["e_max"], [[31.622777], [31.622777]])
    assert_allclose(table["ref_dnde"], [[3.162278e-14], [3.162278e-14]], rtol=1e-5)
    assert_allclose(table["ref_flux"], [[9.683772e-13], [9.683772e-13]], rtol=1e-5)
    assert_allclose(table["ref_eflux"], [[3.453878e-12], [3.453878e-12]], rtol=1e-5)
    assert_allclose(table["stat"], [[16.824042], [17.391981]], rtol=1e-5)
    assert_allclose(table["norm"], [[0.911963], [0.9069318]], rtol=1e-2)
    assert_allclose(table["norm_err"], [[0.057769], [0.057835]], rtol=1e-2)
    assert_allclose(table["counts"], [[[791, np.nan]], [[np.nan, 784]]])
    assert_allclose(table["norm_errp"], [[0.058398], [0.058416]], rtol=1e-2)
    assert_allclose(table["norm_errn"], [[0.057144], [0.057259]], rtol=1e-2)
    assert_allclose(table["norm_ul"], [[1.029989], [1.025061]], rtol=1e-2)
    assert_allclose(table["sqrt_ts"], [[19.384781], [19.161769]], rtol=1e-2)
    assert_allclose(table["ts"], [[375.769735], [367.173374]], rtol=1e-2)
    assert_allclose(table[0]["norm_scan"], [[0.2, 1.0, 5.0]])
    assert_allclose(
        table[0]["stat_scan"],
        [[224.058304, 19.074405, 2063.75636]],
        rtol=1e-5,
    )

    # TODO: fix reference model I/O
    fp = FluxPoints.from_table(
        table=table, format="lightcurve", reference_model=PowerLawSpectralModel()
    )
    assert fp.norm.geom.axes.names == ["energy", "time"]
    assert fp.counts.geom.axes.names == ["dataset", "energy", "time"]
    assert fp.stat_scan.geom.axes.names == ["norm", "energy", "time"]


@requires_data()
def test_lightcurve_estimator_spectrum_datasets_2_energy_bins():
    # Doing a LC on one hour bin
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T01:00:00"]),
        Time(["2010-01-01T01:00:00", "2010-01-01T02:00:00"]),
    ]

    estimator = LightCurveEstimator(
        energy_edges=[1, 5, 30] * u.TeV,
        norm_n_values=3,
        time_intervals=time_intervals,
        selection_optional="all",
    )
    lightcurve = estimator.run(datasets)
    table = lightcurve.to_table(format="lightcurve")

    assert_allclose(table["time_min"], [55197.0, 55197.041667])
    assert_allclose(table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(table["e_ref"], [[2.238721, 12.589254], [2.238721, 12.589254]])
    assert_allclose(table["e_min"], [[1, 5.011872], [1, 5.011872]])
    assert_allclose(table["e_max"], [[5.011872, 31.622777], [5.011872, 31.622777]])
    assert_allclose(
        table["ref_dnde"],
        [[1.995262e-13, 6.309573e-15], [1.995262e-13, 6.309573e-15]],
        rtol=1e-5,
    )
    assert_allclose(
        table["stat"],
        [[8.234951, 8.30321], [2.037205, 15.300507]],
        rtol=1e-5,
    )
    assert_allclose(
        table["norm"],
        [[0.894723, 0.967419], [0.914283, 0.882351]],
        rtol=1e-2,
    )
    assert_allclose(
        table["norm_err"],
        [[0.065905, 0.121288], [0.06601, 0.119457]],
        rtol=1e-2,
    )
    assert_allclose(
        table["counts"],
        [[[669.0, np.nan], [122.0, np.nan]], [[np.nan, 667.0], [np.nan, 117.0]]],
    )
    assert_allclose(
        table["norm_errp"],
        [[0.06664, 0.124741], [0.066815, 0.122832]],
        rtol=1e-2,
    )
    assert_allclose(
        table["norm_errn"],
        [[0.065176, 0.117904], [0.065212, 0.116169]],
        rtol=1e-2,
    )
    assert_allclose(
        table["norm_ul"],
        [[1.029476, 1.224117], [1.049283, 1.134874]],
        rtol=1e-2,
    )
    assert_allclose(
        table["sqrt_ts"],
        [[16.233236, 10.608376], [16.609784, 9.557339]],
        rtol=1e-2,
    )
    assert_allclose(table[0]["norm_scan"], [[0.2, 1.0, 5.0], [0.2, 1.0, 5.0]])
    assert_allclose(
        table[0]["stat_scan"],
        [[153.880281, 10.701492, 1649.609684], [70.178023, 8.372913, 414.146676]],
        rtol=1e-5,
    )

    # those quantities are currently not part of the table so we test separately
    npred = lightcurve.npred.data.squeeze()
    assert_allclose(
        npred,
        [[[669.36, np.nan], [121.66, np.nan]], [[np.nan, 664.41], [np.nan, 115.09]]],
        rtol=1e-3,
    )

    npred_excess_err = lightcurve.npred_excess_err.data.squeeze()
    assert_allclose(
        npred_excess_err,
        [[[26.80, np.nan], [11.31, np.nan]], [[np.nan, 26.85], [np.nan, 11.14]]],
        rtol=1e-3,
    )

    npred_excess_errp = lightcurve.npred_excess_errp.data.squeeze()
    assert_allclose(
        npred_excess_errp,
        [[[27.11, np.nan], [11.63, np.nan]], [[np.nan, 27.15], [np.nan, 11.46]]],
        rtol=1e-3,
    )

    npred_excess_errn = lightcurve.npred_excess_errn.data.squeeze()
    assert_allclose(
        npred_excess_errn,
        [[[26.50, np.nan], [11.00, np.nan]], [[np.nan, 26.54], [np.nan, 10.84]]],
        rtol=1e-3,
    )

    npred_excess_ul = lightcurve.npred_excess_ul.data.squeeze()
    assert_allclose(
        npred_excess_ul,
        [[[418.68, np.nan], [114.19, np.nan]], [[np.nan, 426.74], [np.nan, 105.86]]],
        rtol=1e-3,
    )

    fp = FluxPoints.from_table(
        table=table, format="lightcurve", reference_model=PowerLawSpectralModel()
    )
    assert fp.norm.geom.axes.names == ["energy", "time"]
    assert fp.counts.geom.axes.names == ["dataset", "energy", "time"]
    assert fp.stat_scan.geom.axes.names == ["norm", "energy", "time"]


@requires_data()
def test_lightcurve_estimator_spectrum_datasets_with_mask_fit():
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
        dataset.mask_fit = geom.energy_mask(
            energy_min=energy_min_fit, energy_max=energy_max_fit
        )

    selection = ["scan"]
    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV,
        norm_n_values=3,
        time_intervals=time_intervals,
        selection_optional=selection,
    )
    lightcurve = estimator.run(datasets)
    table = lightcurve.to_table(format="lightcurve")
    assert_allclose(table["time_min"], [55197.0, 55197.041667])
    assert_allclose(table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(table["stat"], [[6.603043], [0.421051]], rtol=1e-3)
    assert_allclose(table["norm"], [[0.885124], [0.967054]], rtol=1e-3)


@requires_data()
def test_lightcurve_estimator_spectrum_datasets_default():
    # Test default time interval: each time interval is equal to the gti of each
    # dataset, here one hour
    datasets = get_spectrum_datasets()
    selection = ["scan"]
    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV, norm_n_values=3, selection_optional=selection
    )
    lightcurve = estimator.run(datasets)
    table = lightcurve.to_table(format="lightcurve")
    assert_allclose(table["time_min"], [55197.0, 55197.041667])
    assert_allclose(table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(table["norm"], [[0.911963], [0.906931]], rtol=1e-3)


@requires_data()
def test_lightcurve_estimator_spectrum_datasets_notordered():
    # Test that if the time intervals given are not ordered in time, it is first ordered
    # correctly and then compute as expected
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
    table = lightcurve.to_table(format="lightcurve")
    assert_allclose(table["time_min"], [55197.0, 55197.041667])
    assert_allclose(table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(table["norm"], [[0.911963], [0.906931]], rtol=1e-3)


@requires_data()
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
    table = lightcurve.to_table(format="lightcurve")

    assert_allclose(table["time_min"], [55197.0])
    assert_allclose(table["time_max"], [55197.083333])
    assert_allclose(table["e_ref"][0], [5.623413])
    assert_allclose(table["e_min"][0], [1])
    assert_allclose(table["e_max"][0], [31.622777])
    assert_allclose(table["ref_dnde"][0], [3.162278e-14], rtol=1e-5)
    assert_allclose(table["ref_flux"][0], [9.683772e-13], rtol=1e-5)
    assert_allclose(table["ref_eflux"][0], [3.453878e-12], rtol=1e-5)
    assert_allclose(table["stat"][0], [34.219808], rtol=1e-5)
    assert_allclose(table["norm"][0], [0.909646], rtol=1e-5)
    assert_allclose(table["norm_err"][0], [0.040874], rtol=1e-3)
    assert_allclose(table["ts"][0], [742.939324], rtol=1e-4)


@requires_data()
def test_lightcurve_estimator_spectrum_datasets_emptybin():
    # Test all dataset in a single LC bin, here two hours
    datasets = get_spectrum_datasets()
    time_intervals = [
        Time(["2010-01-01T00:00:00", "2010-01-01T02:00:00"]),
        Time(["2010-02-01T00:00:00", "2010-02-01T02:00:00"]),
    ]
    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV,
        norm_n_values=3,
        time_intervals=time_intervals,
        selection_optional=["scan"],
    )
    lightcurve = estimator.run(datasets)
    table = lightcurve.to_table(format="lightcurve")

    assert_allclose(table["time_min"], [55197.0])
    assert_allclose(table["time_max"], [55197.083333])


@requires_data()
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

    model = dataset_1.models["source"].copy(name="test_source")
    bkg_model_1 = FoVBackgroundModel(dataset_name="dataset_1")
    bkg_model_2 = FoVBackgroundModel(dataset_name="dataset_2")

    dataset_1.models = [model, bkg_model_1]
    dataset_2.models = [model, bkg_model_2]

    return [dataset_1, dataset_2]


@requires_data()
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
    table = lightcurve.to_table(format="lightcurve")
    assert_allclose(table["time_min"], [55197.0, 55197.041667])
    assert_allclose(table["time_max"], [55197.041667, 55197.083333])
    assert_allclose(table["e_ref"], [[10.857111], [10.857111]])
    assert_allclose(table["e_min"], [[1.178769], [1.178769]], rtol=1e-5)
    assert_allclose(table["e_max"], [[100], [100]])
    assert_allclose(table["ref_dnde"], [[8.483429e-14], [8.483429e-14]], rtol=1e-5)
    assert_allclose(table["ref_flux"], [[8.383429e-12], [8.383429e-12]], rtol=1e-5)
    assert_allclose(table["ref_eflux"], [[4.4407e-11], [4.4407e-11]], rtol=1e-5)
    assert_allclose(table["stat"], [[9402.778975], [9517.750207]], rtol=1e-2)
    assert_allclose(table["norm"], [[0.971592], [0.963286]], rtol=1e-2)
    assert_allclose(table["norm_err"], [[0.044643], [0.044475]], rtol=1e-2)
    assert_allclose(table["sqrt_ts"], [[35.880361], [35.636547]], rtol=1e-2)
    assert_allclose(table["ts"], [[1287.4003], [1269.963491]], rtol=1e-2)

    datasets = get_map_datasets()

    time_intervals2 = [Time(["2010-01-01T00:00:00", "2010-01-01T02:00:00"])]
    estimator2 = LightCurveEstimator(
        energy_edges=[1, 100] * u.TeV,
        source="test_source",
        time_intervals=time_intervals2,
        selection_optional=["scan"],
    )
    lightcurve2 = estimator2.run(datasets)
    table = lightcurve2.to_table(format="lightcurve")
    assert_allclose(table["time_min"][0], [55197.0])
    assert_allclose(table["time_max"][0], [55197.083333])
    assert_allclose(table["e_ref"][0], [10.857111], rtol=1e-5)
    assert_allclose(table["e_min"][0], [1.178769], rtol=1e-5)
    assert_allclose(table["e_max"][0], [100])
    assert_allclose(table["ref_dnde"][0], [8.483429e-14], rtol=1e-5)
    assert_allclose(table["ref_flux"][0], [8.383429e-12], rtol=1e-5)
    assert_allclose(table["ref_eflux"][0], [4.4407e-11], rtol=1e-5)
    assert_allclose(table["stat"][0], [18920.54651], rtol=1e-2)
    assert_allclose(table["norm"][0], [0.967438], rtol=1e-2)
    assert_allclose(table["norm_err"][0], [0.031508], rtol=1e-2)
    assert_allclose(table["counts"][0], [[2205, 2220]])
    assert_allclose(table["ts"][0], [2557.346464], rtol=1e-2)


@requires_data()
def test_recompute_ul():
    datasets = get_spectrum_datasets()
    selection = ["all"]
    estimator = LightCurveEstimator(
        energy_edges=[1, 3, 30] * u.TeV, selection_optional=selection, n_sigma_ul=2
    )
    lightcurve = estimator.run(datasets)
    assert_allclose(
        lightcurve.dnde_ul.data[0], [[[3.260703e-13]], [[1.159354e-14]]], rtol=1e-3
    )

    new_lightcurve = lightcurve.recompute_ul(n_sigma_ul=4)
    assert_allclose(
        new_lightcurve.dnde_ul.data[0], [[[3.774561e-13]], [[1.374421e-14]]], rtol=1e-3
    )
    assert new_lightcurve.meta["n_sigma_ul"] == 4

    # test if scan is not present
    selection = []
    estimator = LightCurveEstimator(
        energy_edges=[1, 30] * u.TeV, selection_optional=selection
    )
    lightcurve = estimator.run(datasets)
    with pytest.raises(ValueError):
        lightcurve.recompute_ul(n_sigma_ul=4)

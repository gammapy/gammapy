# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Column, Table
from astropy.time import Time
from gammapy.data import GTI
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpDecayTemporalModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@pytest.fixture()
def test_meta_table(dataset):
    meta_table = dataset.meta_table
    assert meta_table["TELESCOP"] == "CTA"
    assert meta_table["OBS_ID"] == "0001"
    assert meta_table["INSTRUME"] == "South_Z20_50h"


@pytest.fixture()
def dataset():
    path = "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
    table = Table.read(make_path(path))
    table["e_ref"] = table["e_ref"].quantity.to("TeV")
    gti = GTI.create(start=0 * u.s, stop=30 * u.min)
    data = FluxPoints.from_table(table, format="gadf-sed")
    data.gti = gti
    model = SkyModel(
        spectral_model=PowerLawSpectralModel(
            index=2.3, amplitude="2e-13 cm-2 s-1 TeV-1", reference="1 TeV"
        )
    )
    obs_table = Table()
    obs_table["TELESCOP"] = ["CTA"]
    obs_table["OBS_ID"] = ["0001"]
    obs_table["INSTRUME"] = ["South_Z20_50h"]
    dataset = FluxPointsDataset(model, data, meta_table=obs_table)
    return dataset


@requires_data()
def test_flux_point_dataset_serialization(tmp_path):
    path = "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
    table = Table.read(make_path(path))
    table["e_ref"] = table["e_ref"].quantity.to("TeV")
    data = FluxPoints.from_table(table, format="gadf-sed")

    spectral_model = PowerLawSpectralModel(
        index=2.3, amplitude="2e-13 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model = SkyModel(spectral_model=spectral_model, name="test_model")
    dataset = FluxPointsDataset(model, data, name="test_dataset")

    dataset2 = FluxPointsDataset.read(path, name="test_dataset2")
    assert_allclose(dataset.data.dnde.data, dataset2.data.dnde.data)
    assert dataset.mask_safe.data == dataset2.mask_safe.data
    assert dataset2.name == "test_dataset2"

    Datasets([dataset]).write(
        filename=tmp_path / "tmp_datasets.yaml",
        filename_models=tmp_path / "tmp_models.yaml",
    )

    datasets = Datasets.read(
        filename=tmp_path / "tmp_datasets.yaml",
        filename_models=tmp_path / "tmp_models.yaml",
    )

    new_dataset = datasets[0]
    assert_allclose(new_dataset.data.dnde, dataset.data.dnde, 1e-4)
    if dataset.mask_fit is None:
        assert np.all(new_dataset.mask_fit == dataset.mask_safe)
    assert np.all(new_dataset.mask_safe == dataset.mask_safe)
    assert new_dataset.name == "test_dataset"


@requires_data()
def test_flux_point_dataset_str(dataset):
    assert "FluxPointsDataset" in str(dataset)
    # check print if no models present
    dataset.models = None
    assert "FluxPointsDataset" in str(dataset)


@requires_data()
def test_flux_point_dataset_flux_pred(dataset):

    assert_allclose(dataset.flux_pred()[0].value, 0.00022766, rtol=1e-2)
    dataset.models[0].temporal_model = ExpDecayTemporalModel(
        t0=5.0 * u.hr, t_ref=51543.5 * u.d
    )
    assert_allclose(dataset.flux_pred()[0].value, 0.000472, rtol=1e-3)


def test_flux_point_dataset_creation():
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

    flux_points = FluxPoints.from_table(table=table, format="lightcurve")
    with pytest.raises(ValueError):
        FluxPointsDataset(data=flux_points)


@requires_data()
class TestFluxPointFit:
    def test_fit_pwl_minuit(self, dataset):
        fit = Fit()
        result = fit.run(dataset)
        self.assert_result(result, dataset.models)

    @requires_dependency("sherpa")
    def test_fit_pwl_sherpa(self, dataset):
        fit = Fit(backend="sherpa", optimize_opts={"method": "simplex"})
        result = fit.optimize(datasets=[dataset])
        self.assert_result(result, dataset.models)

    @staticmethod
    def assert_result(result, models):
        assert result.success
        assert_allclose(result.total_stat, 25.2059, rtol=1e-3)

        index = models.parameters["index"]
        assert_allclose(index.value, 2.216, rtol=1e-3)

        amplitude = models.parameters["amplitude"]
        assert_allclose(amplitude.value, 2.1616e-13, rtol=1e-3)

        reference = models.parameters["reference"]
        assert_allclose(reference.value, 1, rtol=1e-8)

    @staticmethod
    def test_stat_profile(dataset):
        fit = Fit()
        result = fit.run(datasets=dataset)

        model = dataset.models[0].spectral_model

        assert_allclose(model.amplitude.error, 1.9e-14, rtol=1e-2)

        model.amplitude.scan_n_values = 3
        model.amplitude.scan_n_sigma = 1
        model.amplitude.interp = "lin"

        profile = fit.stat_profile(
            datasets=dataset,
            parameter="amplitude",
        )

        ts_diff = profile["stat_scan"] - result.total_stat
        assert_allclose(
            model.amplitude.scan_values, [1.97e-13, 2.16e-13, 2.35e-13], rtol=1e-2
        )
        assert_allclose(ts_diff, [110.244116, 0.0, 110.292074], rtol=1e-2, atol=1e-7)

        value = model.parameters["amplitude"].value
        err = model.parameters["amplitude"].error

        model.amplitude.scan_values = np.array([value - err, value, value + err])
        profile = fit.stat_profile(
            datasets=dataset,
            parameter="amplitude",
        )

        ts_diff = profile["stat_scan"] - result.total_stat
        assert_allclose(
            model.amplitude.scan_values, [1.97e-13, 2.16e-13, 2.35e-13], rtol=1e-2
        )
        assert_allclose(ts_diff, [110.244116, 0.0, 110.292074], rtol=1e-2, atol=1e-7)

    @staticmethod
    def test_fp_dataset_plot_fit(dataset):

        with mpl_plot_check():
            dataset.plot_fit(kwargs_residuals=dict(method="diff/model"))

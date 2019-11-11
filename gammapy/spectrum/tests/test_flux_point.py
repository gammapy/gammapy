# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from gammapy.catalog.fermi import SourceCatalog3FGL
from gammapy.modeling import Datasets, Fit
from gammapy.modeling.models import (
    ConstantSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    SkyModels,
    SpectralModel,
)
from gammapy.spectrum import FluxPoints, FluxPointsDataset
from gammapy.utils.testing import (
    assert_quantity_allclose,
    mpl_plot_check,
    requires_data,
    requires_dependency,
)

FLUX_POINTS_FILES = [
    "diff_flux_points.ecsv",
    "diff_flux_points.fits",
    "flux_points.ecsv",
    "flux_points.fits",
]


class LWTestModel(SpectralModel):
    @staticmethod
    def evaluate(x):
        return 1e4 * np.exp(-6 * x)

    def integral(self, xmin, xmax, **kwargs):
        return -1.0 / 6 * 1e4 * (np.exp(-6 * xmax) - np.exp(-6 * xmin))

    def inverse(self, y):
        return -1.0 / 6 * np.log(y * 1e-4)


class XSqrTestModel(SpectralModel):
    @staticmethod
    def evaluate(x):
        return x ** 2

    def integral(self, xmin, xmax, **kwargs):
        return 1.0 / 3 * (xmax ** 3 - xmin ** 2)

    def inverse(self, y):
        return np.sqrt(y)


class ExpTestModel(SpectralModel):
    @staticmethod
    def evaluate(x):
        return np.exp(x * u.Unit("1 / TeV"))

    def integral(self, xmin, xmax, **kwargs):
        return np.exp(xmax * u.Unit("1 / TeV")) - np.exp(xmin * u.Unit("1 / TeV"))

    def inverse(self, y):
        return np.log(y * u.TeV) * u.TeV


def test_e_ref_lafferty():
    """
    Tests Lafferty & Wyatt x-point method.

    Using input function g(x) = 10^4 exp(-6x) against
    check values from paper Lafferty & Wyatt. Nucl. Instr. and Meth. in Phys.
    Res. A 355 (1995) 541-547, p. 542 Table 1
    """
    # These are the results from the paper
    desired = np.array([0.048, 0.190, 0.428, 0.762])

    model = LWTestModel()
    e_min = np.array([0.0, 0.1, 0.3, 0.6])
    e_max = np.array([0.1, 0.3, 0.6, 1.0])
    actual = FluxPoints._e_ref_lafferty(model, e_min, e_max)
    assert_allclose(actual, desired, atol=1e-3)


def test_dnde_from_flux():
    """Tests y-value normalization adjustment method.
    """
    e_min = np.array([10, 20, 30, 40])
    e_max = np.array([20, 30, 40, 50])
    flux = np.array([42, 52, 62, 72])  # 'True' integral flux in this test bin

    # Get values
    model = XSqrTestModel()
    e_ref = FluxPoints._e_ref_lafferty(model, e_min, e_max)
    dnde = FluxPoints._dnde_from_flux(
        flux, model, e_ref, e_min, e_max, pwl_approx=False
    )

    # Set up test case comparison
    dnde_model = model(e_ref)

    # Test comparison result
    desired = model.integral(e_min, e_max)
    # Test output result
    actual = flux * (dnde_model / dnde)
    # Compare
    assert_allclose(actual, desired, rtol=1e-6)


@pytest.mark.parametrize("method", ["table", "lafferty", "log_center"])
def test_compute_flux_points_dnde_exp(method):
    """
    Tests against analytical result or result from gammapy.spectrum.powerlaw.
    """
    model = ExpTestModel()

    e_min = [1.0, 10.0] * u.TeV
    e_max = [10.0, 100.0] * u.TeV

    table = Table()
    table.meta["SED_TYPE"] = "flux"
    table["e_min"] = e_min
    table["e_max"] = e_max

    flux = model.integral(e_min, e_max)
    table["flux"] = flux

    if method == "log_center":
        e_ref = np.sqrt(e_min * e_max)
    elif method == "table":
        e_ref = [2.0, 20.0] * u.TeV
        table["e_ref"] = e_ref
    elif method == "lafferty":
        e_ref = FluxPoints._e_ref_lafferty(model, e_min, e_max)

    result = FluxPoints(table).to_sed_type("dnde", model=model, method=method)

    # Test energy
    actual = result.e_ref
    assert_quantity_allclose(actual, e_ref, rtol=1e-8)

    # Test flux
    actual = result.table["dnde"].quantity
    desired = model(e_ref)
    assert_quantity_allclose(actual, desired, rtol=1e-8)


@pytest.fixture(params=FLUX_POINTS_FILES, scope="session")
def flux_points(request):
    path = "$GAMMAPY_DATA/tests/spectrum/flux_points/" + request.param
    return FluxPoints.read(path)


@pytest.fixture(scope="session")
def flux_points_likelihood():
    path = "$GAMMAPY_DATA/tests/spectrum/flux_points/binlike.fits"
    return FluxPoints.read(path).to_sed_type("dnde")


@requires_data()
class TestFluxPoints:
    def test_info(self, flux_points):
        info = str(flux_points)
        assert flux_points.sed_type in info

    def test_e_ref(self, flux_points):
        actual = flux_points.e_ref
        if flux_points.sed_type == "dnde":
            pass
        elif flux_points.sed_type == "flux":
            desired = np.sqrt(flux_points.e_min * flux_points.e_max)
            assert_quantity_allclose(actual, desired)

    def test_e_min(self, flux_points):
        if flux_points.sed_type == "dnde":
            pass
        elif flux_points.sed_type == "flux":
            actual = flux_points.e_min
            desired = 299530.97 * u.MeV
            assert_quantity_allclose(actual.sum(), desired)

    def test_e_max(self, flux_points):
        if flux_points.sed_type == "dnde":
            pass
        elif flux_points.sed_type == "flux":
            actual = flux_points.e_max
            desired = 399430.975 * u.MeV
            assert_quantity_allclose(actual.sum(), desired)

    def test_write_fits(self, tmp_path, flux_points):
        flux_points.write(tmp_path / "tmp.fits")
        actual = FluxPoints.read(tmp_path / "tmp.fits")
        assert str(flux_points) == str(actual)

    def test_write_ecsv(self, tmp_path, flux_points):
        flux_points.write(tmp_path / "flux_points.ecsv")
        actual = FluxPoints.read(tmp_path / "flux_points.ecsv")
        assert str(flux_points) == str(actual)

    def test_drop_ul(self, flux_points):
        flux_points = flux_points.drop_ul()
        assert not np.any(flux_points.is_ul)

    def test_stack(self, flux_points):
        stacked = FluxPoints.stack([flux_points, flux_points])
        assert len(stacked.table) == 2 * len(flux_points.table)
        assert stacked.sed_type == flux_points.sed_type

    @requires_dependency("matplotlib")
    def test_plot(self, flux_points):
        with mpl_plot_check():
            flux_points.plot()

    @requires_dependency("matplotlib")
    def test_plot_likelihood(self, flux_points_likelihood):
        with mpl_plot_check():
            flux_points_likelihood.plot_likelihood()


@requires_data()
def test_compute_flux_points_dnde_fermi():
    """
    Test compute_flux_points_dnde on fermi source.
    """
    fermi_3fgl = SourceCatalog3FGL()
    source = fermi_3fgl["3FGL J0835.3-4510"]
    flux_points = source.flux_points.to_sed_type(
        "dnde", model=source.spectral_model(), method="log_center", pwl_approx=True
    )
    for column in ["dnde", "dnde_errn", "dnde_errp", "dnde_ul"]:
        actual = flux_points.table["e2" + column].quantity
        desired = flux_points.table[column].quantity * flux_points.e_ref ** 2
        assert_quantity_allclose(actual[:-1], desired[:-1], rtol=1e-1)


@pytest.fixture()
def fit(dataset):
    return Fit(dataset)


@pytest.fixture()
def dataset():
    path = "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
    data = FluxPoints.read(path)
    data.table["e_ref"] = data.e_ref.to("TeV")
    model = PowerLawSpectralModel(
        index=2.3, amplitude="2e-13 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    dataset = FluxPointsDataset(model, data)
    return dataset


@requires_data()
def test_flux_point_dataset_serialization(tmp_path):
    path = "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
    data = FluxPoints.read(path)
    data.table["e_ref"] = data.e_ref.to("TeV")
    # TODO: remove duplicate definition this once model is redefine as skymodel
    spatial_model = ConstantSpatialModel()
    spectral_model = PowerLawSpectralModel(
        index=2.3, amplitude="2e-13 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model = SkyModel(spatial_model, spectral_model, name="test_model")
    dataset = FluxPointsDataset(SkyModels([model]), data, name="test_dataset")

    Datasets([dataset]).to_yaml(tmp_path, prefix="tmp")
    datasets = Datasets.from_yaml(
        tmp_path / "tmp_datasets.yaml", tmp_path / "tmp_models.yaml"
    )
    new_dataset = datasets[0]
    assert_allclose(new_dataset.data.table["dnde"], dataset.data.table["dnde"], 1e-4)
    if dataset.mask_fit is None:
        assert np.all(new_dataset.mask_fit == dataset.mask_safe)
    assert np.all(new_dataset.mask_safe == dataset.mask_safe)
    assert new_dataset.name == "test_dataset"


@requires_data()
def test_flux_point_dataset_str(dataset):
    assert "FluxPointsDataset" in str(dataset)


@requires_data()
class TestFluxPointFit:
    @requires_dependency("iminuit")
    def test_fit_pwl_minuit(self, fit):
        optimize_opts = {"backend": "minuit"}
        result = fit.run(optimize_opts=optimize_opts)
        self.assert_result(result)

    @requires_dependency("sherpa")
    def test_fit_pwl_sherpa(self, fit):
        result = fit.optimize(backend="sherpa", method="simplex")
        self.assert_result(result)

    @staticmethod
    def assert_result(result):
        assert result.success
        assert_allclose(result.total_stat, 25.2059, rtol=1e-3)

        index = result.parameters["index"]
        assert_allclose(index.value, 2.216, rtol=1e-3)

        amplitude = result.parameters["amplitude"]
        assert_allclose(amplitude.value, 2.1616e-13, rtol=1e-3)

        reference = result.parameters["reference"]
        assert_allclose(reference.value, 1, rtol=1e-8)

    @staticmethod
    @requires_dependency("iminuit")
    def test_likelihood_profile(fit):
        optimize_opts = {"backend": "minuit"}

        result = fit.run(optimize_opts=optimize_opts)

        profile = fit.likelihood_profile("amplitude", nvalues=3, bounds=1)

        ts_diff = profile["likelihood"] - result.total_stat
        assert_allclose(ts_diff, [110.1, 0, 110.1], rtol=1e-2, atol=1e-7)

        value = result.parameters["amplitude"].value
        err = result.parameters.error("amplitude")
        values = np.array([value - err, value, value + err])

        profile = fit.likelihood_profile("amplitude", values=values)

        ts_diff = profile["likelihood"] - result.total_stat
        assert_allclose(ts_diff, [110.1, 0, 110.1], rtol=1e-2, atol=1e-7)

    @staticmethod
    @requires_dependency("matplotlib")
    def test_fp_dataset_peek(fit):
        fp_dataset = fit.datasets[0]

        with mpl_plot_check():
            fp_dataset.peek(method="diff/model")

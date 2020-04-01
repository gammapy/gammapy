# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from gammapy.catalog.fermi import SourceCatalog3FGL
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import PowerLawSpectralModel, SpectralModel
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
            flux_points_likelihood.plot_ts_profiles()


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


@pytest.fixture(scope="session")
def model():
    return PowerLawSpectralModel()


@pytest.fixture(scope="session")
def flux_points_dnde(model):
    e_ref = [np.sqrt(10), np.sqrt(10 * 100)] * u.TeV
    table = Table()
    table.meta["SED_TYPE"] = "dnde"
    table["e_ref"] = e_ref
    table["dnde"] = model(e_ref)
    return FluxPoints(table)


@pytest.fixture(scope="session")
def flux_points_e2dnde(model):
    e_ref = [np.sqrt(10), np.sqrt(10 * 100)] * u.TeV
    table = Table()
    table.meta["SED_TYPE"] = "e2dnde"
    table["e_ref"] = e_ref
    table["e2dnde"] = (model(e_ref) * e_ref ** 2).to("erg cm-2 s-1")
    return FluxPoints(table)


@pytest.fixture(scope="session")
def flux_points_flux(model):
    e_min = [1, 10] * u.TeV
    e_max = [10, 100] * u.TeV

    table = Table()
    table.meta["SED_TYPE"] = "flux"
    table["e_min"] = e_min
    table["e_max"] = e_max
    table["flux"] = model.integral(e_min, e_max)
    return FluxPoints(table)


def test_dnde_to_e2dnde(flux_points_dnde, flux_points_e2dnde):
    actual = flux_points_dnde.to_sed_type("e2dnde").table
    desired = flux_points_e2dnde.table
    assert_allclose(actual["e2dnde"], desired["e2dnde"])


def test_e2dnde_to_dnde(flux_points_e2dnde, flux_points_dnde):
    actual = flux_points_e2dnde.to_sed_type("dnde").table
    desired = flux_points_dnde.table
    assert_allclose(actual["dnde"], desired["dnde"])


def test_flux_to_dnde(flux_points_flux, flux_points_dnde):
    actual = flux_points_flux.to_sed_type("dnde", method="log_center").table
    desired = flux_points_dnde.table
    assert_allclose(actual["e_ref"], desired["e_ref"])
    assert_allclose(actual["dnde"], desired["dnde"])

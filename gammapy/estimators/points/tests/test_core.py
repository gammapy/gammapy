# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from gammapy.catalog.fermi import SourceCatalog3FGL, SourceCatalog4FGL
from gammapy.estimators import FluxPoints
from gammapy.estimators.map.core import DEFAULT_UNIT
from gammapy.modeling.models import PowerLawSpectralModel, SpectralModel
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import (
    assert_quantity_allclose,
    mpl_plot_check,
    requires_data,
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
        return x**2

    def integral(self, xmin, xmax, **kwargs):
        return 1.0 / 3 * (xmax**3 - xmin**2)

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


def test_energy_ref_lafferty():
    """
    Tests Lafferty & Wyatt x-point method.

    Using input function g(x) = 10^4 exp(-6x) against
    check values from paper Lafferty & Wyatt. Nucl. Instr. and Meth. in Phys.
    Res. A 355 (1995) 541-547, p. 542 Table 1
    """
    # These are the results from the paper
    desired = np.array([0.048, 0.190, 0.428, 0.762])

    model = LWTestModel()
    energy_min = np.array([0.0, 0.1, 0.3, 0.6])
    energy_max = np.array([0.1, 0.3, 0.6, 1.0])
    actual = FluxPoints._energy_ref_lafferty(model, energy_min, energy_max)
    assert_allclose(actual, desired, atol=1e-3)


@pytest.mark.xfail
def test_dnde_from_flux():
    """Tests y-value normalization adjustment method."""
    table = Table()
    table["e_min"] = np.array([10, 20, 30, 40])
    table["e_max"] = np.array([20, 30, 40, 50])
    table["flux"] = np.array([42, 52, 62, 72])  # 'True' integral flux in this test bin

    # Get values
    model = XSqrTestModel()
    table["e_ref"] = FluxPoints._energy_ref_lafferty(
        model, table["e_min"], table["e_max"]
    )
    dnde = FluxPoints.from_table(table, reference_model=model)

    # Set up test case comparison
    dnde_model = model(table["e_ref"])

    # Test comparison result
    desired = model.integral(table["e_min"], table["e_max"])
    # Test output result
    actual = table["flux"] * (dnde_model / dnde)
    # Compare
    assert_allclose(actual, desired, rtol=1e-6)


@pytest.mark.xfail
@pytest.mark.parametrize("method", ["table", "lafferty", "log_center"])
def test_compute_flux_points_dnde_exp(method):
    """
    Tests against analytical result or result from a powerlaw.
    """
    model = ExpTestModel()

    energy_min = [1.0, 10.0] * u.TeV
    energy_max = [10.0, 100.0] * u.TeV

    table = Table()
    table.meta["SED_TYPE"] = "flux"
    table["e_min"] = energy_min
    table["e_max"] = energy_max

    flux = model.integral(energy_min, energy_max)
    table["flux"] = flux

    if method == "log_center":
        energy_ref = np.sqrt(energy_min * energy_max)
    elif method == "table":
        energy_ref = [2.0, 20.0] * u.TeV
    elif method == "lafferty":
        energy_ref = FluxPoints._energy_ref_lafferty(model, energy_min, energy_max)

    table["e_ref"] = energy_ref

    result = FluxPoints.from_table(table, reference_model=model)

    # Test energy
    actual = result.energy_ref
    assert_quantity_allclose(actual, energy_ref, rtol=1e-8)

    # Test flux
    actual = result.dnde
    desired = model(energy_ref)
    assert_quantity_allclose(actual, desired, rtol=1e-8)


@requires_data()
def test_fermi_to_dnde():

    catalog_4fgl = SourceCatalog4FGL("$GAMMAPY_DATA/catalogs/fermi/gll_psc_v20.fit.gz")
    src = catalog_4fgl["FGES J1553.8-5325"]
    fp = src.flux_points

    assert_allclose(
        fp.dnde.quantity[1, 0, 0],
        4.567393e-10 * u.Unit("cm-2 s-1 MeV-1"),
        rtol=1e-5,
    )


@pytest.fixture(params=FLUX_POINTS_FILES, scope="session")
def flux_points(request):
    path = "$GAMMAPY_DATA/tests/spectrum/flux_points/" + request.param
    return FluxPoints.read(path)


@pytest.fixture(scope="session")
def flux_points_likelihood():
    path = "$GAMMAPY_DATA/tests/spectrum/flux_points/binlike.fits"
    return FluxPoints.read(path)


@requires_data()
class TestFluxPoints:
    def test_info(self, flux_points):
        info = str(flux_points)
        assert "geom" in info
        assert "axes" in info
        assert "ref. model" in info
        assert "quantities" in info

    def test_energy_ref(self, flux_points):
        actual = flux_points.energy_ref
        desired = np.sqrt(flux_points.energy_min * flux_points.energy_max)
        assert_quantity_allclose(actual, desired)

    def test_energy_min(self, flux_points):
        actual = flux_points.energy_min
        desired = 299530.97 * u.MeV
        assert_quantity_allclose(actual.sum(), desired)

    def test_energy_max(self, flux_points):
        actual = flux_points.energy_max
        desired = 399430.975 * u.MeV
        assert_quantity_allclose(actual.sum(), desired)

    def test_write_fits(self, tmp_path, flux_points):
        flux_points.write(tmp_path / "tmp.fits", sed_type=flux_points.sed_type_init)
        actual = FluxPoints.read(tmp_path / "tmp.fits")
        actual._data.pop("is_ul", None)
        flux_points._data.pop("is_ul", None)
        assert str(flux_points) == str(actual)

    def test_write_ecsv(self, tmp_path, flux_points):
        flux_points.write(
            tmp_path / "flux_points.ecsv", sed_type=flux_points.sed_type_init
        )
        actual = FluxPoints.read(tmp_path / "flux_points.ecsv")
        actual._data.pop("is_ul", None)
        flux_points._data.pop("is_ul", None)
        assert str(flux_points) == str(actual)

    def test_quantity_access(self, flux_points_likelihood):
        assert flux_points_likelihood.sqrt_ts
        assert flux_points_likelihood.ts
        assert flux_points_likelihood.stat
        assert_allclose(flux_points_likelihood.n_sigma_ul, 2)
        assert flux_points_likelihood.sed_type_init == "likelihood"

    def test_plot(self, flux_points):

        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        ax.xaxis.set_units(u.eV)

        yunit = DEFAULT_UNIT[flux_points.sed_type_init]
        ax.yaxis.set_units(yunit)

        with mpl_plot_check():
            flux_points.plot(ax=ax)

    def test_plot_likelihood(self, flux_points_likelihood):
        with mpl_plot_check():
            flux_points_likelihood.plot_ts_profiles()

    def test_plot_likelihood_error(self, flux_points_likelihood):
        del flux_points_likelihood._data["stat_scan"]

        with pytest.raises(AttributeError):
            ax = plt.subplot()
            flux_points_likelihood.plot_ts_profiles(ax=ax)


@requires_data()
def test_flux_points_single_bin_dnde():
    path = make_path("$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits")
    table = Table.read(path)

    table_single_bin = table[1:2]
    fp = FluxPoints.from_table(table_single_bin, sed_type="dnde")

    with pytest.raises(ValueError):
        _ = fp.flux_ref

    with mpl_plot_check():
        fp.plot(sed_type="e2dnde")

    with pytest.raises(ValueError):
        fp.to_table(sed_type="flux")

    table = fp.to_table(sed_type="dnde")

    assert_allclose(table["e_ref"], 153.992 * u.MeV, rtol=0.001)
    assert "e_min" not in table.colnames
    assert "e_max" not in table.colnames


@requires_data()
def test_compute_flux_points_dnde_fermi():
    """
    Test compute_flux_points_dnde on fermi source.
    """
    fermi_3fgl = SourceCatalog3FGL()
    source = fermi_3fgl["3FGL J0835.3-4510"]
    flux_points = source.flux_points
    table = source.flux_points_table

    for column in ["e2dnde", "e2dnde_errn", "e2dnde_errp", "e2dnde_ul"]:
        actual = table[column].quantity
        desired = getattr(flux_points, column).quantity.squeeze()
        assert_quantity_allclose(actual[:-1], desired[:-1], rtol=0.05)


@requires_data()
def test_plot_fp_no_ul():
    path = make_path("$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits")
    table = Table.read(path)
    table.remove_column("dnde_ul")
    fp = FluxPoints.from_table(table, sed_type="dnde")

    with mpl_plot_check():
        fp.plot()


@requires_data()
def test_is_ul(tmp_path):
    catalog_4fgl = SourceCatalog4FGL("$GAMMAPY_DATA/catalogs/fermi/gll_psc_v20.fit.gz")
    src = catalog_4fgl["FGES J1553.8-5325"]
    fp = src.flux_points

    is_ul = fp._data["is_ul"].data.squeeze()

    assert_allclose(fp.is_ul.data.squeeze(), is_ul)
    table = fp.to_table()
    assert_allclose(table["is_ul"].data.data, is_ul)

    fp.sqrt_ts_threshold_ul = 100
    assert_allclose(fp.is_ul.data.squeeze(), np.ones(is_ul.shape, dtype=bool))
    table = fp.to_table()
    assert_allclose(table["is_ul"].data.data, np.ones(is_ul.shape, dtype=bool))

    table.write(tmp_path / "test_modif_ul_threshold.fits")
    table_read = Table.read(tmp_path / "test_modif_ul_threshold.fits")
    assert_allclose(table_read["is_ul"].data.data, np.ones(is_ul.shape, dtype=bool))
    fp_read = FluxPoints.from_table(table_read)
    assert_allclose(fp_read.is_ul.data.squeeze(), np.ones(is_ul.shape, dtype=bool))
    assert_allclose(fp_read.to_table()["is_ul"], fp_read.is_ul.data.squeeze())

    fp.is_ul = is_ul
    assert_allclose(fp.is_ul.data.squeeze(), is_ul)
    table = fp.to_table()
    assert_allclose(table["is_ul"].data.data, is_ul)


def test_flux_points_plot_no_error_bar():
    table = Table()
    pwl = PowerLawSpectralModel()
    e_ref = np.geomspace(1, 100, 7) * u.TeV

    table["e_ref"] = e_ref
    table["dnde"] = pwl(e_ref)
    table.meta["SED_TYPE"] = "dnde"

    flux_points = FluxPoints.from_table(table)
    with mpl_plot_check():
        _ = flux_points.plot(sed_type="dnde")

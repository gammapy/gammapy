# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
import pytest
from astropy.table import Table
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ...utils.energy import EnergyBounds
from ..models import PowerLaw, ConstantModel
from .. import SpectrumObservation, SpectrumFitResult, FluxPoints, SpectrumResult


@pytest.fixture(scope="session")
def fit_result():
    filename = "$GAMMAPY_EXTRA/datasets/joint-crab/spectra/hess/pha_obs23592.fits"
    obs = SpectrumObservation.read(filename)
    best_fit_model = PowerLaw(
        index=2.1, amplitude=1e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    )
    npred = obs.predicted_counts(best_fit_model).data.data.value
    covar = np.diag([0.1 ** 2, 1e-12 ** 2, 0])
    best_fit_model.parameters.covariance = covar
    fit_range = [0.1, 50] * u.TeV
    return SpectrumFitResult(
        model=best_fit_model,
        fit_range=fit_range,
        statname="wstat",
        statval=42,
        npred=npred,
        obs=obs,
    )


@pytest.fixture(scope="session")
def flux_points():
    energies = EnergyBounds.equal_log_spacing(1, 10, 3, unit="TeV")
    dnde = [1e-11, 2e-11, 0.5e-11, 6e-11] * u.Unit("cm-2 s-1 TeV-1")
    dnde_err = np.ones(len(energies)) * 1e-12 * u.Unit("cm-2 s-1 TeV-1")
    dnde_ul = [np.nan, np.nan, np.nan, 6e-11] * u.Unit("cm-2 s-1 TeV-1")
    is_ul = [False, False, False, True]
    table = Table(
        data=[energies, dnde, dnde_err, is_ul, dnde_ul],
        names=["e_ref", "dnde", "dnde_err", "is_ul", "dnde_ul"],
        meta={"SED_TYPE": "dnde"},
    )
    return FluxPoints(table)


@pytest.fixture(scope="session")
def spectrum_result(flux_points):
    model = ConstantModel(const=1e-11 * u.Unit("cm-2 s-1 TeV-1"))
    model.parameters.set_error(0, 1e-10 * u.Unit("cm-2 s-1 TeV-1"))
    return SpectrumResult(model=model, points=flux_points)


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestSpectrumFitResult:
    @requires_dependency("uncertainties")
    def test_basic(self, fit_result):
        assert "PowerLaw" in str(fit_result)
        assert "index" in fit_result.to_table().colnames

    @requires_dependency("yaml")
    def test_io(self, tmpdir, fit_result):
        filename = tmpdir / "test.yaml"
        fit_result.to_yaml(filename)
        read_result = SpectrumFitResult.from_yaml(filename)
        test_e = 12.5 * u.TeV
        assert_quantity_allclose(fit_result.model(test_e), read_result.model(test_e))

    @requires_dependency("matplotlib")
    def test_plot(self, fit_result):
        with mpl_plot_check():
            fit_result.plot()


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestSpectrumResult:
    def test_basic(self, spectrum_result):
        assert "SpectrumResult" in str(spectrum_result)

    def test_residuals(self, spectrum_result):
        res, res_err = spectrum_result.flux_point_residuals
        assert_quantity_allclose(res, [0, 1, -0.5, np.nan])
        assert_quantity_allclose(res_err, [0.1, 0.1, 0.1, np.nan])

    @requires_dependency("matplotlib")
    @requires_dependency("uncertainties")
    def test_plot(self, spectrum_result):
        with mpl_plot_check():
            spectrum_result.plot(energy_range=[1, 10] * u.TeV, energy_power=2)

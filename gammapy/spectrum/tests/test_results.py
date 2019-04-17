# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u
from astropy.table import Table
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import EnergyBounds
from ..models import PowerLaw
from .. import SpectrumObservation, SpectrumFitResult, FluxPoints


@pytest.fixture(scope="session")
def fit_result():
    filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23592.fits"
    obs = SpectrumObservation.read(filename)
    best_fit_model = PowerLaw(
        index=2.1, amplitude=1e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    )
    npred = obs.predicted_counts(best_fit_model).data.data.value
    covariance = np.diag([0.1 ** 2, 1e-12 ** 2, 0])
    best_fit_model.parameters.covariance = covariance
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


@requires_data("gammapy-data")
class TestSpectrumFitResult:
    @requires_dependency("uncertainties")
    def test_basic(self, fit_result):
        assert "PowerLaw" in str(fit_result)

    def test_io(self, tmpdir, fit_result):
        filename = tmpdir / "test.yaml"
        fit_result.to_yaml(filename)
        read_result = SpectrumFitResult.from_yaml(filename)
        test_e = 12.5 * u.TeV
        assert_quantity_allclose(fit_result.model(test_e), read_result.model(test_e))

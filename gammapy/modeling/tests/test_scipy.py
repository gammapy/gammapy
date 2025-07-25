# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.scipy import (
    confidence_scipy,
    optimize_scipy,
    stat_profile_ul_scipy,
)
from gammapy.estimators import FluxPoints
from gammapy.utils.testing import requires_data


class MyDataset:
    def __init__(self, parameters):
        self.parameters = parameters

    def fcn(self):
        x, y, z = [p.value for p in self.parameters]
        x_opt, y_opt, z_opt = 2, 3e5, 4e-5
        x_err, y_err, z_err = 0.2, 3e4, 4e-6
        return (
            ((x - x_opt) / x_err) ** 2
            + ((y - y_opt) / y_err) ** 2
            + ((z - z_opt) / z_err) ** 2
        )


@pytest.fixture()
def pars():
    x = Parameter("x", 2.1)
    y = Parameter("y", 3.1e5, scale=1e5)
    z = Parameter("z", 4.1e-5, scale=1e-5)
    return Parameters([x, y, z])


@pytest.mark.parametrize("method", ["Nelder-Mead", "L-BFGS-B", "Powell", "BFGS"])
def test_scipy_basic(pars, method):
    ds = MyDataset(pars)
    factors, info, optimizer = optimize_scipy(
        function=ds.fcn, parameters=pars, method=method
    )

    assert info["success"]
    assert_allclose(ds.fcn(), 0, atol=1e-5)

    # Check the result in parameters is OK
    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e5, rtol=1e-3)
    assert_allclose(pars["z"].value, 4e-5, rtol=2e-2)


@pytest.mark.parametrize("method", ["Nelder-Mead", "L-BFGS-B", "Powell"])
def test_scipy_frozen(pars, method):
    ds = MyDataset(pars)
    pars["y"].frozen = True

    factors, info, optimizer = optimize_scipy(
        function=ds.fcn, parameters=pars, method=method
    )

    assert info["success"]

    assert_allclose(pars["x"].value, 2, rtol=1e-4)
    assert_allclose(pars["y"].value, 3.1e5)
    assert_allclose(pars["z"].value, 4.0e-5, rtol=1e-4)
    assert_allclose(ds.fcn(), 0.1111111, rtol=1e-5)


@pytest.mark.parametrize("method", ["L-BFGS-B"])
def test_scipy_limits(pars, method):
    ds = MyDataset(pars)
    pars["y"].min = 301000

    factors, info, minuit = optimize_scipy(
        function=ds.fcn, parameters=pars, method=method
    )

    assert info["success"]

    # Check the result in parameters is OK
    assert_allclose(pars["x"].value, 2, rtol=1e-2)
    assert_allclose(pars["y"].value, 301000, rtol=1e-3)


def test_scipy_confidence(pars):
    ds = MyDataset(pars)
    factors, info, _ = optimize_scipy(function=ds.fcn, parameters=pars)

    assert_allclose(ds.fcn(), 0, atol=1e-5)

    par = pars["x"]
    par.scan_min, par.scan_max = 0, 10

    result = confidence_scipy(function=ds.fcn, parameters=pars, parameter=par, sigma=1)

    assert result["success_errp"]
    assert result["success_errn"]

    assert_allclose(result["errp"], 0.2, rtol=1e-3)
    assert_allclose(result["errn"], 0.2, rtol=1e-3)


@requires_data()
def test_stat_profile_ul_scipy():
    # Test normal profile with a minima
    x = np.linspace(-5, 5, 7)
    y = x**2
    ul = stat_profile_ul_scipy(x, y)
    assert_allclose(ul, 2)

    ul = stat_profile_ul_scipy(x, y, interp_scale="lin")
    assert_allclose(ul, 1.9111111)

    # Test with real data
    flux_point = FluxPoints.read(
        "$GAMMAPY_DATA/estimators/pks2155_hess_lc/pks2155_hess_lc.fits",
        format="lightcurve",
    )
    value_scan = flux_point.stat_scan.geom.axes["norm"].center
    stat_scan = np.sum(flux_point.stat_scan.data, axis=0).ravel()
    ul = stat_profile_ul_scipy(value_scan, stat_scan, delta_ts=4)
    assert_allclose(ul, 1.1123425)

    # Test a flat profile i.e. all minima
    x = np.linspace(0, 10, 10)
    y = np.ones_like(x) * 5

    with pytest.raises(
        ValueError,
        match="Statistic profile is flat therefore no best-fit value can be determined.",
    ):
        stat_profile_ul_scipy(x, y)

    # Test with NaN result
    x = np.linspace(0, 5, 20)
    y = (x - 2.5) ** 2 + 10

    with pytest.raises(
        RuntimeError, match="Failed to find upper limit: no valid root found."
    ):
        stat_profile_ul_scipy(x, y, delta_ts=20)

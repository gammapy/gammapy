# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
import astropy.units as u
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency
from ...irf import EffectiveAreaTable, EnergyDispersion
from ...spectrum import integrate_spectrum, SpectrumEvaluator
from ..models import ExponentialCutoffPowerLaw, PowerLaw, TableModel, PowerLaw2


def test_integrate_spectrum():
    """
    Test numerical integration against analytical solution.
    """
    emin = Quantity(1, "TeV")
    emax = Quantity(10, "TeV")
    pwl = PowerLaw(index=2.3)

    ref = pwl.integral(emin=emin, emax=emax)

    val = integrate_spectrum(pwl, emin, emax)
    assert_quantity_allclose(val, ref)


@requires_dependency("uncertainties")
def test_integrate_spectrum_ecpl():
    """
    Test ecpl integration. Regression test for
    https://github.com/gammapy/gammapy/issues/687
    """
    ecpl = ExponentialCutoffPowerLaw(
        index=2.3,
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1 * u.TeV,
        lambda_=0.1 / u.TeV,
    )
    ecpl.parameters.set_parameter_errors(
        {"index": 0.2, "amplitude": 1e-13 * u.Unit("cm-2 s-1 TeV-1")}
    )
    emin, emax = 1 * u.TeV, 1e10 * u.TeV
    res = ecpl.integral_error(emin, emax)

    assert res.unit == "cm-2 s-1"
    assert_allclose(res.value, [5.95657824e-13, 9.27830251e-14], rtol=1e-5)


def get_test_cases():
    e_true = Quantity(np.logspace(-1, 2, 120), "TeV")
    e_reco = Quantity(np.logspace(-1, 2, 100), "TeV")
    return [
        dict(model=PowerLaw(amplitude="1e2 TeV-1"), e_true=e_true, npred=999),
        dict(
            model=PowerLaw2(amplitude="1", emin="0.1 TeV", emax="100 TeV"),
            e_true=e_true,
            npred=1,
        ),
        dict(
            model=PowerLaw(amplitude="1e-11 TeV-1 cm-2 s-1"),
            aeff=EffectiveAreaTable.from_parametrization(e_true),
            livetime="10 h",
            npred=1448.05960,
        ),
        dict(
            model=PowerLaw(reference="1 GeV", amplitude="1e-11 GeV-1 cm-2 s-1"),
            aeff=EffectiveAreaTable.from_parametrization(e_true),
            livetime="30 h",
            npred=4.34417881,
        ),
        dict(
            model=PowerLaw(amplitude="1e-11 TeV-1 cm-2 s-1"),
            aeff=EffectiveAreaTable.from_parametrization(e_true),
            edisp=EnergyDispersion.from_gauss(
                e_reco=e_reco, e_true=e_true, bias=0, sigma=0.2
            ),
            livetime="10 h",
            npred=1437.450076,
        ),
        dict(
            model=TableModel(
                energy=[0.1, 0.2, 0.3, 0.4] * u.TeV,
                values=[4.0, 3.0, 1.0, 0.1] * u.Unit("TeV-1"),
            ),
            e_true=[0.1, 0.2, 0.3, 0.4] * u.TeV,
            npred=0.554513062,
        ),
    ]


@pytest.mark.parametrize("case", get_test_cases())
def test_counts_predictor(case):
    opts = case.copy()
    del opts["npred"]
    predictor = SpectrumEvaluator(**opts)
    actual = predictor.compute_npred().total_counts
    assert_allclose(actual, case["npred"])

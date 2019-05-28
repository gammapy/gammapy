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
from ..powerlaw import power_law_energy_flux, power_law_evaluate, power_law_flux
from ..models import ExponentialCutoffPowerLaw, PowerLaw, TableModel


def test_integrate_spectrum():
    """
    Test numerical integration against analytical solution.
    """
    e1 = Quantity(1, "TeV")
    e2 = Quantity(10, "TeV")
    einf = Quantity(1e10, "TeV")
    e = Quantity(1, "TeV")
    g = 2.3
    I = Quantity(1e-12, "cm-2 s-1")

    ref = power_law_energy_flux(I=I, g=g, e=e, e1=e1, e2=e2)
    norm = power_law_flux(I=I, g=g, e=e, e1=e1, e2=einf)
    f = lambda x: x * power_law_evaluate(x, norm, g, e)
    val = integrate_spectrum(f, e1, e2)
    assert_quantity_allclose(val, ref)

    # Test quantity handling
    e2_ = Quantity(1e4, "GeV")
    val_ = integrate_spectrum(f, e1, e2_)
    assert_quantity_allclose(val, val_)


@requires_dependency("uncertainties")
def test_integrate_spectrum_uncertainties():
    """
    Test numerical integration against analytical solution.
    """
    from uncertainties import unumpy

    e1 = 1.0
    e2 = 10.0
    einf = 1e10
    e = 1.0
    g = unumpy.uarray(2.3, 0.2)
    I = unumpy.uarray(1e-12, 1e-13)

    ref = power_law_energy_flux(I=I, g=g, e=e, e1=e1, e2=e2)
    norm = power_law_flux(I=I, g=g, e=e, e1=e1, e2=einf)
    f = lambda x: x * power_law_evaluate(x, norm, g, e)
    val = integrate_spectrum(f, e1, e2)

    assert_allclose(unumpy.nominal_values(val), unumpy.nominal_values(ref))
    assert_allclose(unumpy.std_devs(val), unumpy.std_devs(ref))


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
    actual = predictor.compute_npred().total_counts.value
    assert_allclose(actual, case["npred"])

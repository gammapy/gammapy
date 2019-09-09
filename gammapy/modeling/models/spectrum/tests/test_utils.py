# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.units import Quantity
from gammapy.modeling.models import ExponentialCutoffPowerLaw, PowerLaw
from gammapy.modeling.models.spectrum.utils import integrate_spectrum
from gammapy.utils.testing import assert_quantity_allclose, requires_dependency


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

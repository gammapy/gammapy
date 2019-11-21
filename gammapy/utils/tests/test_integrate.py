# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.units import Quantity
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.utils.integrate import integrate_spectrum
from gammapy.utils.testing import assert_quantity_allclose


def test_integrate_spectrum():
    """
    Test numerical integration against analytical solution.
    """
    emin = Quantity(1, "TeV")
    emax = Quantity(10, "TeV")
    pwl = PowerLawSpectralModel(index=2.3)

    ref = pwl.integral(emin=emin, emax=emax)

    val = integrate_spectrum(pwl, emin, emax)
    assert_quantity_allclose(val, ref)

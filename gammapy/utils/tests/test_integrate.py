# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.units import Quantity
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.utils.integrate import trapz_loglog
from gammapy.utils.testing import assert_quantity_allclose


def test_trapz_loglog():
    energy = Quantity([1, 10], "TeV")
    pwl = PowerLawSpectralModel(index=2.3)

    ref = pwl.integral(emin=energy[0], emax=energy[1])

    val = trapz_loglog(pwl(energy), energy)
    assert_quantity_allclose(val, ref)

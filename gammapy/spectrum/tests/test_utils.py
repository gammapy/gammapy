# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...spectrum import (
    LogEnergyAxis,
    integrate_spectrum,
)
from ..powerlaw import power_law_energy_flux, power_law_evaluate, power_law_flux


@requires_dependency('scipy')
def test_LogEnergyAxis():
    from scipy.stats import gmean
    energy = Quantity([1, 10, 100], 'TeV')
    energy_axis = LogEnergyAxis(energy)

    assert_allclose(energy_axis.x, [0, 1, 2])
    assert_quantity_allclose(energy_axis.energy, energy)

    energy = Quantity(gmean([1, 10]), 'TeV')
    pix = energy_axis.world2pix(energy.to('MeV'))
    assert_allclose(pix, 0.5)

    world = energy_axis.pix2world(pix)
    assert_quantity_allclose(world, energy)


def test_integrate_spectrum():
    """
    Test numerical integration against analytical solution.
    """
    e1 = Quantity(1, 'TeV')
    e2 = Quantity(10, 'TeV')
    einf = Quantity(1E10, 'TeV')
    e = Quantity(1, 'TeV')
    g = 2.3
    I = Quantity(1E-12, 'cm-2 s-1')

    ref = power_law_energy_flux(I=I, g=g, e=e, e1=e1, e2=e2)
    norm = power_law_flux(I=I, g=g, e=e, e1=e1, e2=einf)
    f = lambda x: x * power_law_evaluate(x, norm, g, e)
    val = integrate_spectrum(f, e1, e2)
    assert_quantity_allclose(val, ref)


@requires_dependency('uncertainties')
def test_integrate_spectrum():
    """
    Test numerical integration against analytical solution.
    """
    from uncertainties import unumpy
    e1 = 1.
    e2 = 10.
    einf = 1E10
    e = 1.
    g = unumpy.uarray(2.3, 0.2)
    I = unumpy.uarray(1E-12, 1E-13)

    ref = power_law_energy_flux(I=I, g=g, e=e, e1=e1, e2=e2)
    norm = power_law_flux(I=I, g=g, e=e, e1=e1, e2=einf)
    f = lambda x: x * power_law_evaluate(x, norm, g, e)
    val = integrate_spectrum(f, e1, e2)

    assert_allclose(unumpy.nominal_values(val), unumpy.nominal_values(ref))
    assert_allclose(unumpy.std_devs(val), unumpy.std_devs(ref))

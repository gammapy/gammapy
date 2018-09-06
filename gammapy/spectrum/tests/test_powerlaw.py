# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.units import Quantity
from ...utils.testing import assert_quantity_allclose
from ..powerlaw import (
    power_law_pivot_energy,
    power_law_energy_flux,
    power_law_integral_flux,
    power_law_compatibility,
)


def test_one():
    """Test one case"""
    I = power_law_integral_flux(f=1, g=2)

    assert_allclose(I, 1)


def test_powerlaw_energy_flux():
    """
    Test energy flux computation for power law against numerical solution.
    """
    e1 = Quantity(1, "TeV")
    e2 = Quantity(10, "TeV")
    e = Quantity(1, "TeV")
    g = 2.3
    I = Quantity(1e-12, "cm-2 s-1")

    val = power_law_energy_flux(I=I, g=g, e=e, e1=e1, e2=e2)

    ref = Quantity(2.1615219876151536e-12, "TeV cm-2 s-1")
    assert_quantity_allclose(val, ref)


def test_e_pivot():
    """Hard-coded example from fit example in survey/spectra.
    """
    e_pivot = power_law_pivot_energy(
        energy_ref=1, f0=5.35510540e-11, d_gamma=0.0318377, cov=6.56889442e-14
    )

    assert_allclose(e_pivot, 3.3540034240210987)


def test_compatibility():
    """
    Run a test case with hardcoded numbers:

    HESS J1912+101
    1FGL 1913.7+1007c

    We use the following symbols and units:
    e = pivot energy (MeV)
    f = flux density (cm^-2 s^-1 MeV^-1)
    g = "gamma" = spectral index
    """
    # Fermi power-law parameters
    e_fermi = 1296.2734
    f_fermi = 3.791907E-12
    f_err_fermi = 5.6907235E-13
    g_fermi = 2.3759267
    g_err_fermi = 0.08453985
    par_fermi = (e_fermi, f_fermi, f_err_fermi, g_fermi, g_err_fermi)

    # HESS power-law parameters
    e_hess = 1e6
    f_hess = 3.5 * 1e-12 * 1e-6
    f_err_hess = 0.6 * 1e-12 * 1e-6
    g_hess = 2.2
    g_err_hess = 0.2
    par_hess = (e_hess, f_hess, f_err_hess, g_hess, g_err_hess)

    compatibility = power_law_compatibility(par_fermi, par_hess)

    # Note: I just put the numbers here, didn't verify them!
    assert_allclose(compatibility["g_match"], 2.0901127509816506)
    assert_allclose(compatibility["sigma_low"], -3.380819211512078)
    assert_allclose(compatibility["sigma_high"], -0.5494362450917478)
    assert_allclose(compatibility["sigma_combined"], 3.4251742624791612)

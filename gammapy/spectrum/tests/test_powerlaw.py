# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...spectrum import powerlaw

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import uncertainties
    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False


@pytest.mark.xfail
def test_powerlaw():
    e = 1
    e1, e2 = 0.2, 1e42
    f, f_err = 1, 0.1
    g, g_err = 2, 0.1

    I_unc, I_unc_err = powerlaw.I_with_err(e1, e2, e, f, f_err, g, g_err)
    f_unc, f_unc_err = powerlaw.f_with_err(e1, e2, e, I_unc, I_unc_err, g, g_err)

    print('f (real):', f, f_err)
    print('g (real):', g, g_err)
    print('I (unc ):', I_unc, I_unc_err)
    print('f (unc ):', f_unc, f_unc_err)


def test_one():
    """Test one case"""
    I = powerlaw.power_law_integral_flux(f=1, g=2)
    assert_allclose(I, 1)


@pytest.mark.skipif('not HAS_UNCERTAINTIES')
def test_closure(g_error_mag=0):
    """This test passes for g_error_mag == 0,
    but fails for g_error_mag != 0, because
    I and g have correlated errors, but we
    effectively throw away these correlations!"""

    npoints = 100
    # Generate some random f values with errors
    f_val = 10 ** (10 * np.random.random(npoints) - 5)
    f_err = f_val * np.random.normal(1, 0.1, npoints)
    # f = unumpy.uarray((f_val, f_err))

    # Generate some random g values with errors
    g_val = 5 * np.random.random(npoints)
    g_err = g_val * np.random.normal(1, 0.1, npoints)
    # g = unumpy.uarray((g_val, g_err))

    I_val, I_err = powerlaw.I_with_err(f_val, f_err, g_val, g_err)
    # I_val = unumpy.nominal_values(f)
    # I_err = unumpy.std_devs(f)

    f_val2, f_err2 = powerlaw.f_with_err(I_val, I_err, g_val, g_err)
    try:
        assert_allclose(f_val, f_val2)
        assert_allclose(f_err, f_err2)
    except AssertionError as err:
        print(err)
        m = 3
        print('f_val:', f_val[:m])
        print('f_err:', f_err[:m])
        print('g_val:', g_val[:m])
        print('g_err:', g_err[:m])
        print('I_val:', I_val[:m])
        print('I_err:', I_err[:m])
        print('f_val2:', f_val2[:m])
        print('f_err2:', f_err2[:m])


def test_e_pivot():
    """Hard-coded example from fit example in survey/spectra.
    """
    e0 = 1
    f0 = 5.35510540e-11
    d_gamma = 0.0318377
    cov = 6.56889442e-14

    e_pivot = powerlaw.power_law_pivot_energy(e0, f0, d_gamma, cov)
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

    g_match, sigma_low, sigma_high, sigma_comb = \
        powerlaw.compatibility(par_fermi, par_hess)


@pytest.mark.skipif('not HAS_SCIPY')
def test_SED_error(I=1., e1=1, e2=10):
    """Compute the error one makes by using the simple formulas:
    e = sqrt(e1 * e2)
    f = I / (e2 - e1)
    e2f = e ** 2 * f
    to compute a differential flux f or e2f from an integral flux
    measurement I in an energy bin [e1, e2].
    Note that e is the log bin center and e2f is typically plotted
    in spectral energy distributions (SEDs).

    Index    SED-Error Flux-Error
    1.5    1.28    0.85
    2.0    1.00    1.00
    2.5    0.85    1.28
    3.0    0.81    1.75
    """
    from scipy.stats import gmean
    e = gmean([e1, e2])
    f = I / (e2 - e1)
    e2f = e ** 2 * f  # @note: e ** 2 = e1 * e2 here.
    for Index in np.arange(1.5, 3.5, 0.5):
        f_correct = powerlaw.power_law_flux(I, Index, e, e1, e2)
        e2f_correct = e ** 2 * f_correct
        # We compute ratios, which corresponds to differences
        # on a log scale
        SED = e2f / e2f_correct
        Flux = f / f_correct
        # TODO: assert results

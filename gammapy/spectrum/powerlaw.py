# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Power law spectrum helper functions.

* Not part of the public Gammapy API!
* To be used only within Gammapy, and only in cases where the
  ``PowerLaw`` and ``PowerLaw2`` classes in ``gammapy.spectrum.models``
  are too slow or don't contain some functionality.
* The reason to keep this as standalone functions is speed, i.e.
  no spectral model instantiation takes place, and there's
  no handling of quantities here.
* We might want to completely remove this module and expose the functionality
  directly on the spectral model classes some day.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = [
    "power_law_evaluate",
    "power_law_pivot_energy",
    "power_law_df_over_f",
    "power_law_flux",
    "power_law_energy_flux",
    "power_law_integral_flux",
    "power_law_g_from_f",
    "power_law_g_from_points",
    "power_law_I_from_points",
    "power_law_f_from_points",
    "power_law_f_with_err",
    "power_law_I_with_err",
    "power_law_compatibility",
]

E_INF = 1e10
"""
Practically infinitely high energy.

Whoa!
"""

g_DEFAULT = 2
"""Default spectral index.

Use this if you don't know a better one.
:-)
"""


def power_law_evaluate(energy, norm, gamma, energy_ref):
    r"""Differential flux at a given energy.

    .. math:: f(energy) = N (E / E_0) ^ - \Gamma

    with norm ``N``, energy ``E``, reference energy ``E0`` and spectral index :math:`\Gamma`.

    Parameters
    ----------
    energy : array_like
        Energy at which to compute the differential flux
    gamma : array_like
        Power law spectral index
    """
    return norm * (energy / energy_ref) ** (-gamma)


def power_law_pivot_energy(energy_ref, f0, d_gamma, cov):
    """Compute pivot (a.k.a. decorrelation) energy.

    Defined as smallest df / f.

    Reference: http://arxiv.org/pdf/0910.4881
    """
    return energy_ref * np.exp(cov / (f0 * d_gamma ** 2))


def power_law_df_over_f(e, e0, f0, df0, dg, cov):
    """Compute relative flux error at any given energy.

    Used to draw butterflies.

    Reference: http://arxiv.org/pdf/0910.4881 Equation (1)
    """
    term1 = (df0 / f0) ** 2
    term2 = 2 * cov / f0 * np.log(e / e0)
    term3 = (dg * np.log(e / e0)) ** 2
    return np.sqrt(term1 - term2 + term3)


def _power_law_conversion_factor(g, e, e1, e2):
    """Conversion factor between differential and integral flux."""
    term1 = e / (-g + 1)
    term2 = (e2 / e) ** (-g + 1) - (e1 / e) ** (-g + 1)
    return term1 * term2


def power_law_flux(I=1, g=g_DEFAULT, e=1, e1=1, e2=E_INF):
    """Compute differential flux for a given integral flux.

    Parameters
    ----------
    I : array_like
        Integral flux in ``energy_min``, ``energy_max`` band
    g : array_like
        Power law spectral index
    e : array_like
        Energy at which to compute the differential flux
    e1 : array_like
        Energy band minimum
    e2 : array_like
        Energy band maximum

    Returns
    -------
    flux : `numpy.array`
        Differential flux at ``energy``.
    """
    return I / _power_law_conversion_factor(g, e, e1, e2)


def power_law_energy_flux(I, g=g_DEFAULT, e=1, e1=1, e2=10):
    r"""
    Compute energy flux between e1 and e2 for a given integral flux.

    The analytical solution for the powerlaw case is given by:

    .. math::

        G(E_1, E_2) = I(\epsilon, \infty) \, \frac{1-\Gamma}
        {2-\Gamma} \, \frac{E_1^{2-\Gamma} - E_2^{2-\Gamma}}{\epsilon^{1-\Gamma}}

    Parameters
    ----------
    I : array_like
        Integral flux in ``energy_min``, ``energy_max`` band
    g : array_like
        Power law spectral index
    e : array_like
        Energy at above which the integral flux is given.
    e1 : array_like
        Energy band minimum
    e2 : array_like
        Energy band maximum
    """
    g1 = 1. - g
    g2 = 2. - g
    factor = g1 / g2 * (e1 ** g2 - e2 ** g2) / e ** g1
    return I * factor


def power_law_integral_flux(f=1, g=g_DEFAULT, e=1, e1=1, e2=E_INF):
    """Compute integral flux for a given differential flux.

    Parameters
    ----------
    f : array_like
        Differential flux at ``energy``
    g : array_like
        Power law spectral index
    e : array_like
        Energy at which the differential flux is given
    e1 : array_like
        Energy band minimum
    e2 : array_like
        Energy band maximum

    Returns
    -------
    flux : `numpy.array`
        Integral flux in ``energy_min``, ``energy_max`` band
    """
    return f * _power_law_conversion_factor(g, e, e1, e2)


def power_law_g_from_f(e, f, de=1):
    """Spectral index at a given energy e for a given function f(e)"""
    e1, e2 = e, e + de
    f1, f2 = f(e1), f(e2)
    return power_law_g_from_points(e1, e2, f1, f2)


def power_law_g_from_points(e1, e2, f1, f2):
    """Spectral index for two given differential flux points"""
    return -np.log(f2 / f1) / np.log(e2 / e1)


def power_law_I_from_points(e1, e2, f1, f2):
    """Integral flux in energy bin for power law"""
    g = power_law_g_from_points(e1, e2, f1, f2)
    pl_int_flux = f1 * e1 / (-g + 1) * ((e2 / e1) ** (-g + 1) - 1)
    return pl_int_flux


def power_law_f_from_points(e1, e2, f1, f2, e):
    """Linear interpolation"""
    logdy = np.log(f2 / f1)
    logdx = np.log(e2 / e1)
    logy = np.log(f1) + np.log(e / e1) * (logdy / logdx)
    return np.exp(logy)


def power_law_f_with_err(
    I_val=1, I_err=0, g_val=g_DEFAULT, g_err=0, e=1, e1=1, e2=E_INF
):
    """Wrapper for f so the user doesn't have to know about
    the uncertainties module"""
    from uncertainties import unumpy

    I = unumpy.uarray(I_val, I_err)
    g = unumpy.uarray(g_val, g_err)
    _f = power_law_flux(I, g, e, e1, e2)
    f_val = unumpy.nominal_values(_f)
    f_err = unumpy.std_devs(_f)
    return f_val, f_err


def power_law_I_with_err(
    f_val=1, f_err=0, g_val=g_DEFAULT, g_err=0, e=1, e1=1, e2=E_INF
):
    """Wrapper for f so the user doesn't have to know about
    the uncertainties module"""
    from uncertainties import unumpy

    f = unumpy.uarray(f_val, f_err)
    g = unumpy.uarray(g_val, g_err)
    _I = power_law_integral_flux(f, g, e, e1, e2)
    I_val = unumpy.nominal_values(_I)
    I_err = unumpy.std_devs(_I)
    return I_val, I_err


def power_law_compatibility(par_low, par_high):
    """Quantify compatibility of power-law measurements in two energy bands.

    Reference: 2008ApJ...679.1299F Equation (2)

    Compute spectral compatibility parameters for the
    situation where two power laws were measured in a low
    and a high spectral energy band.
    par_low and par_high are the measured parameters,
    which must be lists in the following order:
    e, f, f_err, g, g_err
    where e is the pivot energy, f is the flux density
    and g the spectral index
    """
    # Unpack power-law parameters
    e_high, f_high, f_err_high, g_high, g_err_high = par_high
    e_low, f_low, f_err_low, g_low, g_err_low = par_low

    log_delta_e = np.log10(e_high) - np.log10(e_low)
    log_delta_f = np.log10(f_high) - np.log10(f_low)
    # g_match is the index obtained by connecting the two points
    # with a power law, i.e. a straight line in the log_e, log_f plot
    g_match = -log_delta_f / log_delta_e

    # sigma is the number of standard deviations the match index
    # is different from the measured index in one band.
    # (see Funk et al. (2008ApJ...679.1299F) eqn. 2)
    sigma_low = (g_match - g_low) / g_err_low
    sigma_high = (g_match - g_high) / g_err_high
    sigma_combined = np.sqrt(sigma_low ** 2 + sigma_high ** 2)

    return {
        "g_match": g_match,
        "sigma_low": sigma_low,
        "sigma_high": sigma_high,
        "sigma_combined": sigma_combined,
    }

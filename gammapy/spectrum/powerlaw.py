# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Power law spectrum helper functions.

Convert differential and integral fluxes with error propagation.
"""
import numpy as np

__all__ = ['diff_flux', 'e_pivot', 'df_over_f', 'f', 'I',
           'g_from_f', 'g_from_points', 'I_from_points',
           'f_from_points', 'f_with_err', 'I_with_err', 'compatibility']

E_INF = 1e10  # practically infinitely high flux
g_DEFAULT = 2


def diff_flux(energy, norm, gamma, eref):
    """Differential flux at a given energy"""
    return norm * (energy / eref) ** (-gamma)


def e_pivot(e0, f0, d_gamma, cov):
    """Compute pivot (a.k.a. decorrelation) energy.

    Defined as smallest df / f.
    Reference: http://arxiv.org/pdf/0910.4881
    """
    result = e0 * np.exp(cov / (f0 * d_gamma ** 2))
    print('e0 = %s, f0 = %s, d_gamma = %s, cov = %s, e_pivot = %s' %
          (e0, f0, d_gamma, cov, result))
    return result


def df_over_f(e, e0, f0, df0, dg, cov):
    """Compute relative flux error at any given energy.
    Used to draw butterflies.
    Reference: http://arxiv.org/pdf/0910.4881 Equation (1)
    """
    term1 = (df0 / f0) ** 2
    term2 = 2 * cov / f0 * np.log(e / e0)
    term3 = (dg * np.log(e / e0)) ** 2
    return np.sqrt(term1 - term2 + term3)


def _conversion_factor(g, e, e1, e2):
    """Conversion factor between differential and integral flux"""
    g = np.abs(g)
    return e / (-g + 1) * ((e2 / e) ** (-g + 1) - (e1 / e) ** (-g + 1))


def f(I=1, g=g_DEFAULT, e=1, e1=1, e2=E_INF):
    """Differential flux f at energy e for a given integral
    flux I in energy band e1 to e2 and spectral index g"""
    return I / _conversion_factor(g, e, e1, e2)


def I(f=1, g=g_DEFAULT, e=1, e1=1, e2=E_INF):
    """Integral flux I in energy band e1 to e2 for a given
    differential flux f at energy e and spectral index g"""
    return f * _conversion_factor(g, e, e1, e2)


def g_from_f(e, f, de=1):
    """Spectral index at a given energy e for a given function f(e)"""
    e1, e2 = e, e + de
    f1, f2 = f(e1), f(e2)
    return g_from_points(e1, e2, f1, f2)


def g_from_points(e1, e2, f1, f2):
    """Spectral index for two given differential flux points"""
    return -np.log(f2 / f1) / np.log(e2 / e1)


def I_from_points(e1, e2, f1, f2):
    """Integral flux in energy bin for power law"""
    g = g_from_points(e1, e2, f1, f2)
    pl_int_flux = (f1 * e1 / (-g + 1) *
                   ((e2 / e1) ** (-g + 1) - 1))
    return pl_int_flux


def f_from_points(e1, e2, f1, f2, e):
    """Linear interpolation"""
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    f1 = np.asarray(f1, float)
    f2 = np.asarray(f2, float)
    e = np.asarray(e, float)

    logdy = np.log(f2 / f1)
    logdx = np.log(e2 / e1)
    logy = np.log(f1) + np.log(e / e1) * (logdy / logdx)
    return np.exp(logy)


def f_with_err(I_val=1, I_err=0, g_val=g_DEFAULT, g_err=0,
               e=1, e1=1, e2=E_INF):
    """Wrapper for f so the user doesn't have to know about
    the uncertainties module"""
    from uncertainties import unumpy
    I = unumpy.uarray(I_val, I_err)
    g = unumpy.uarray(g_val, g_err)
    _f = f(I, g, e, e1, e2)
    f_val = unumpy.nominal_values(_f)
    f_err = unumpy.std_devs(_f)
    return f_val, f_err


def I_with_err(f_val=1, f_err=0, g_val=g_DEFAULT, g_err=0,
               e=1, e1=1, e2=E_INF):
    """Wrapper for f so the user doesn't have to know about
    the uncertainties module"""
    from uncertainties import unumpy
    f = unumpy.uarray(f_val, f_err)
    g = unumpy.uarray(g_val, g_err)
    _I = I(f, g, e, e1, e2)
    I_val = unumpy.nominal_values(_I)
    I_err = unumpy.std_devs(_I)
    return I_val, I_err


def compatibility(par_low, par_high):
    """
    Quantify spectral compatibility of power-law
    measurements in two energy bands.

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
    # Unpack power-law paramters
    e_high, f_high, f_err_high, g_high, g_err_high = par_high
    e_low, f_low, f_err_low, g_low, g_err_low = par_low

    log_delta_e = np.log10(e_high) - np.log10(e_low)
    log_delta_f = np.log10(f_high) - np.log10(f_low)
    # g_match is the index obtained by connecting the two points
    # with a power law, i.e. a straight line in the log_e, log_f plot
    g_match = -log_delta_f / log_delta_e

    # sigma is the number of standar deviations the match index
    # is different from the measured index in one band.
    # (see Funk et al. (2008ApJ...679.1299F) eqn. 2)
    sigma_low = (g_match - g_low) / g_err_low
    sigma_high = (g_match - g_high) / g_err_high
    sigma_comb = np.sqrt(sigma_low ** 2 + sigma_high ** 2)

    return g_match, sigma_low, sigma_high, sigma_comb

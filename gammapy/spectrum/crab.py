# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Published Crab nebula reference spectra.
========================================

The Crab is often used as a standard candle in gamma-ray astronomy.
Statements like "this source has a flux of 10 % Crab" or
"our sensitivity is 2 % Crab" are common.

Here we provide a reference of what the Crab flux actually is according
to several publications.

Unless noted otherwise:
diff_flux @ 1 TeV in units cm^-2 s^-1 TeV^-1
int_flux > 1 TeV in units cm^-2 s^-1
"""
import numpy as np
from astropy.units import Unit

__all__ = ['diff_flux', 'int_flux', 'spectral_index', 'to_crab']

# HESS publication: 2006A&A...457..899A
hess_pl = {'diff_flux': 3.45e-11,
           'index': 2.63,
           'int_flux': 2.26e-11}
# Note that for ecpl, the diff_flux is not
# the differential flux at 1 TeV, that you
# get by multiplying with exp(-e / cutoff)
hess_ecpl = {'diff_flux': 3.76e-11,
             'index': 2.39,
             'cutoff': 14.3,
             'int_flux': 2.27e-11}
# HEGRA publication : 2004ApJ...614..897A
hegra = {'diff_flux': 2.83e-11,
         'index': 2.62,
         'int_flux': 1.75e-11}
# Meyer et al. publication: 2010arXiv1008.4524M
# diff_flux and index were calculated numerically
# by hand at 1 TeV as a finite differene
meyer = {'diff_flux': 3.3457e-11,
         'index': 2.5362,
         'int_flux': 2.0744e-11}
refs = {'meyer': meyer,
        'hegra': hegra,
        'hess_pl': hess_pl,
        'hess_ecpl': hess_ecpl}

DEFAULT_REF = 'meyer'


def meyer(energy, energyFlux=False):
    """Differential Crab flux at a given energy (TeV).

    erg cm^-2 s^-1 if energyFlux=True
    cm^-1 s^-1 TeV^-1 if energyFlux=False

    @see: Meyer et al., 2010arXiv1008.4524M, Appendix D
    """
    p = np.array([-0.00449161, 0, 0.0473174,
                  - 0.179475, -0.53616, -10.2708])
    log_e = np.log10(np.asarray(energy))
    log_f = np.poly1d(p)(log_e)
    f = 10 ** log_f
    if energyFlux:
        return f
    else:
        return Unit('erg').to('TeV') * f / energy ** 2


def diff_flux(e=1, ref=DEFAULT_REF):
    """Differential Crab flux (cm^-2 s^-1 TeV^-1) at energy e (TeV)
    according to some reference publication
    """
    if ref == 'hegra':
        f = hegra['diff_flux']
        g = hegra['index']
        return f * e ** (-g)
    elif ref == 'hess_pl':
        f = hess_pl['diff_flux']
        g = hess_pl['index']
        return f * e ** (-g)
    elif ref == 'hess_ecpl':
        f = hess_ecpl['diff_flux']
        g = hess_ecpl['index']
        e_c = hess_ecpl['cutoff']
        return f * e ** (-g) * np.exp(-e / e_c)
    elif ref == 'meyer':
        return meyer(e)
    else:
        raise ValueError('Unknown ref: %s' % ref)


def int_flux(e1=1, e2=1e4, ref=DEFAULT_REF):
    """Integral Crab flux (cm^-2 s^-1) in energy band e1 to e2 (TeV)
    according to some reference publication
    """
    # @todo there are integration problems with e2=1e6.
    # test and use the integrator that works in log space!!!
    """
    In [36]: spec.crab.int_flux(0.2, 1e4, ref='hess_ecpl')
[  2.43196827e-10] [  2.61476507e-18]
Out[36]: array([  2.43196827e-10])

In [37]: spec.crab.int_flux(0.2, 1e5, ref='hess_ecpl')
Warning: The algorithm does not converge.  Roundoff error is detected
  in the extrapolation table.  It is assumed that the requested tolerance
  cannot be achieved, and that the returned result (if full_output = 1) is
  the best which can be obtained.
[  2.43283459e-10] [  4.37319063e-10]
Out[37]: array([  2.43283459e-10])

In [38]: spec.crab.int_flux(0.2, 1e6, ref='hess_ecpl')
[  6.40098358e-48] [  1.27271100e-47]
Out[38]: array([  6.40098358e-48])
    """
    from scipy.integrate import quad
    # @todo How does one usually handle 0-dim and 1-dim
    # arrays at the same time?
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    npoints = e1.size
    e1, e2 = e1.reshape(npoints), e2.reshape(npoints)
    I, I_err = np.empty_like(e1), np.empty_like(e2)
    for ii in range(npoints):
        I[ii], I_err[ii] = quad(diff_flux, e1[ii], e2[ii],
                                (ref), epsabs=1e-20)
    return I


def spectral_index(e=1, ref=DEFAULT_REF):
    """Spectral index (positive number) at energy e (TeV)
    """
    # Compute spectral index as slope in log -- log
    # as a finite difference
    eps = 1 + 1e-3
    f1 = diff_flux(e, ref)
    f2 = diff_flux(eps * e, ref)
    return (np.log10(f1) - np.log10(f2)) / np.log10(eps)


def to_crab(f, apply_mask=True, min=1e-3, max=300):
    """Convert to Crab units
    """
    f_crab_ref = int_flux(e1=1, ref='meyer')
    # The 100 is to get % and the 1e4 is to convert sensitivity to cm^-2
    f_crab = 100 * f / 1e4 / f_crab_ref
    if apply_mask:
        # Get rid of invalid values (negative default and 0) and
        # also of very high values
        mask = (f_crab < min) | (f_crab > max)
        f_crab[mask] = np.nan
    return f_crab

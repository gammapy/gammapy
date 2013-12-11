# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Published Crab nebula reference spectra.

The Crab is often used as a standard candle in gamma-ray astronomy.
Statements like "this source has a flux of 10 % Crab" or
"our sensitivity is 2 % Crab" are common.

Here we provide a reference of what the Crab flux actually is according
to several publications.
"""
from __future__ import print_function, division
import numpy as np
from astropy.units import Unit

__all__ = ['crab_flux', 'crab_integral_flux', 'crab_spectral_index', 'convert_to_crab_flux']

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
references = {'meyer': meyer,
        'hegra': hegra,
        'hess_pl': hess_pl,
        'hess_ecpl': hess_ecpl}

DEFAULT_REFERENCE = 'meyer'


def crab_flux(energy=1, reference=DEFAULT_REFERENCE):
    """Differential Crab flux.

    The following published Crab spectra are available:
    
    * 'HEGRA' : 
    * 'HESS PL' and 'HESS ECPL' :
    * 'Meyer' :  2010arXiv1008.4524M, Appendix D
    
    Parameters
    ----------
    energy : array_like
        Energy (TeV)
    reference : {'hegra', 'hess_pl', 'hess_ecpl', 'meyer'}
        Published Crab reference spectrum

    Returns
    -------
    flux : array
        Differential flux (cm^-2 s^-1 TeV^-1)
    """
    if reference == 'hegra':
        f = hegra['diff_flux']
        g = hegra['index']
        return f * energy ** (-g)
    elif reference == 'hess_pl':
        f = hess_pl['diff_flux']
        g = hess_pl['index']
        return f * energy ** (-g)
    elif reference == 'hess_ecpl':
        f = hess_ecpl['diff_flux']
        g = hess_ecpl['index']
        e_c = hess_ecpl['cutoff']
        return f * energy ** (-g) * np.exp(-energy / e_c)
    elif reference == 'meyer':
        # Meyer et al., 2010arXiv1008.4524M, Appendix D
        p = np.array([-0.00449161, 0, 0.0473174,
                      - 0.179475, -0.53616, -10.2708])
        log_e = np.log10(np.asarray(energy))
        log_f = np.poly1d(p)(log_e)
        f = 10 ** log_f
        return Unit('erg').to('TeV') * f / energy ** 2
    else:
        raise ValueError('Unknown reference: %s' % reference)


def crab_integral_flux(e1=1, e2=1e4, reference=DEFAULT_REFERENCE):
    """Integral Crab flux (cm^-2 s^-1) in energy band e1 to e2 (TeV)
    according to some reference publication
    """
    # @todo there are integration problems with e2=1e6.
    # test and use the integrator that works in log space!!!
    """
    In [36]: spec.crab.int_flux(0.2, 1e4, reference='hess_ecpl')
[  2.43196827e-10] [  2.61476507e-18]
Out[36]: array([  2.43196827e-10])

In [37]: spec.crab.int_flux(0.2, 1e5, reference='hess_ecpl')
Warning: The algorithm does not converge.  Roundoff error is detected
  in the extrapolation table.  It is assumed that the requested tolerance
  cannot be achieved, and that the returned result (if full_output = 1) is
  the best which can be obtained.
[  2.43283459e-10] [  4.37319063e-10]
Out[37]: array([  2.43283459e-10])

In [38]: spec.crab.int_flux(0.2, 1e6, reference='hess_ecpl')
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
        I[ii], I_err[ii] = quad(crab_flux, e1[ii], e2[ii],
                                (reference), epsabs=1e-20)
    return I


def crab_spectral_index(e=1, reference=DEFAULT_REFERENCE):
    """Spectral index (positive number) at energy e (TeV)
    """
    # Compute spectral index as slope in log -- log
    # as a finite difference
    eps = 1 + 1e-3
    f1 = crab_flux(e, reference)
    f2 = crab_flux(eps * e, reference)
    return (np.log10(f1) - np.log10(f2)) / np.log10(eps)


def convert_to_crab_flux(f, apply_mask=True, min=1e-3, max=300):
    """Convert a given differential flux to Crab units.
    """
    f_crab_reference = crab_integral_flux(e1=1, reference='meyer')
    # The 100 is to get % and the 1e4 is to convert sensitivity to cm^-2
    f_crab = 100 * f / 1e4 / f_crab_reference
    if apply_mask:
        # Get rid of invalid values (negative default and 0) and
        # also of very high values
        mask = (f_crab < min) | (f_crab > max)
        f_crab[mask] = np.nan
    return f_crab

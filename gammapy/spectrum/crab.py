# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Published Crab nebula reference spectra.

The Crab is often used as a standard candle in gamma-ray astronomy.
Statements like "this source has a flux of 10 % Crab" or
"our sensitivity is 2 % Crab" are common.

Here we provide a reference of what the Crab flux actually is according
to several publications:

* 'HEGRA' : http://adsabs.harvard.edu/abs/2000ApJ...539..317A
* 'HESS PL' and 'HESS ECPL' : http://adsabs.harvard.edu/abs/2006A%26A...457..899A
* 'Meyer' :  http://adsabs.harvard.edu/abs/2010A%26A...523A...2M, Appendix D

"""
from __future__ import print_function, division
import numpy as np
from astropy.units import Unit

__all__ = ['crab_flux', 'crab_integral_flux', 'crab_spectral_index',
           'CRAB_DEFAULT_REFERENCE', 'CRAB_REFERENCES'
           ]

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

CRAB_REFERENCES = ['meyer', 'hegra', 'hess_pl', 'hess_ecpl']

CRAB_DEFAULT_REFERENCE = 'meyer'


def crab_flux(energy=1, reference=CRAB_DEFAULT_REFERENCE):
    """Differential Crab flux.

    See the ``gammapy.spectrum.crab`` module docstring for a description
    of the available reference spectra.

    Parameters
    ----------
    energy : array_like
        Energy (TeV)
    reference : {{'hegra', 'hess_pl', 'hess_ecpl', 'meyer'}}
        Published Crab reference spectrum

    Returns
    -------
    flux : array
        Differential flux (cm^-2 s^-1 TeV^-1) at ``energy``
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
        log_energy = np.log10(np.asarray(energy))
        log_flux = np.poly1d(p)(log_energy)
        flux = 10 ** log_flux
        return Unit('erg').to('TeV') * flux / energy ** 2
    else:
        raise ValueError('Unknown reference: {0}'.format(reference))


def crab_integral_flux(energy_min=1, energy_max=1e4, reference=CRAB_DEFAULT_REFERENCE):
    """Integral Crab flux.

    See the ``gammapy.spectrum.crab`` module docstring for a description
    of the available reference spectra.

    Parameters
    ----------
    energy_min, energy_max : array_like
        Energy band (TeV)
    reference : {{'hegra', 'hess_pl', 'hess_ecpl', 'meyer'}}
        Published Crab reference spectrum

    Returns
    -------
    flux : array
        Integral flux (cm^-2 s^-1) in energy band ``energy_min`` to ``energy_max``
    """
    from scipy.integrate import quad
    # @todo How does one usually handle 0-dim and 1-dim
    # arrays at the same time?
    energy_min, energy_max = np.asarray(energy_min, dtype=float), np.asarray(energy_max, dtype=float)
    npoints = energy_min.size
    energy_min, energy_max = energy_min.reshape(npoints), energy_max.reshape(npoints)
    I, I_err = np.empty_like(energy_min), np.empty_like(energy_max)
    for ii in range(npoints):
        I[ii], I_err[ii] = quad(crab_flux, energy_min[ii], energy_max[ii],
                                (reference), epsabs=1e-20)
    return I


def crab_spectral_index(energy=1, reference=CRAB_DEFAULT_REFERENCE):
    """Spectral index (positive number) at a given energy.

    See the ``gammapy.spectrum.crab`` module docstring for a description
    of the available reference spectra.

    Parameters
    ----------
    energy : array_like
        Energy (TeV)
    reference : {{'hegra', 'hess_pl', 'hess_ecpl', 'meyer'}}
        Published Crab reference spectrum

    Returns
    -------
    spectral_index : array
        Spectral index at ``energy``
    """
    # Compute spectral index as slope in log -- log
    # as a finite difference
    eps = 1 + 1e-3
    f1 = crab_flux(energy, reference)
    f2 = crab_flux(eps * energy, reference)
    return (np.log10(f1) - np.log10(f2)) / np.log10(eps)

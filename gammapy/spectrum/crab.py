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
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from .models import PowerLaw, ExponentialCutoffPowerLaw, SpectralModel
from ..extern.bunch import Bunch

__all__ = [
    'CrabSpectrum',
]

# HESS publication: 2006A&A...457..899A
#'int_flux' = 2.26e-11
hess_pl = {'amplitude': 3.45e-11 * u.Unit('1 / (cm2 s TeV)'),
           'index': 2.63,
           'reference' : 1 * u.TeV}

# Note that for ecpl, the diff_flux is not
# the differential flux at 1 TeV, that you
# get by multiplying with exp(-e / cutoff)
# int_flux: 2.27e-11
hess_ecpl = {'amplitude': 3.76e-11 * u.Unit('1 / (cm2 s TeV)'),
             'index': 2.39,
             'lambda_': 1 / (14.3 * u.TeV),
             'reference': 1 * u.TeV}

# HEGRA publication : 2004ApJ...614..897A
# int_flux': 1.75e-11
hegra = {'amplitude': 2.83e-11 * u.Unit('1 / (cm2 s TeV)'),
         'index': 2.62,
         'reference': 1 * u.TeV}

# Meyer et al. publication: 2010arXiv1008.4524M
# diff_flux and index were calculated numerically
# by hand at 1 TeV as a finite differene
meyer = {'diff_flux': 3.3457e-11 * u.Unit('1 / (cm2 s TeV)'),
         'index': 2.5362,
         'int_flux': 2.0744e-11}


#TODO: make this a general LogPolynomial spectral model
class MeyerCrabModel(SpectralModel):
    """
    Log polynomial model as used by 2010arXiv1008.4524M.
    """
    def __init__(self):
        coefficients = np.array([-0.00449161, 0, 0.0473174, -0.179475,
                                 -0.53616, -10.2708])
        self.parameters = Bunch(coefficients=coefficients)

    @staticmethod
    def evaluate(energy, coefficients):
        polynomial = np.poly1d(coefficients)
        log_energy = np.log10(energy.to('TeV').value)
        log_flux = polynomial(log_energy)
        flux = np.power(10, log_flux) * u.Unit('erg / (cm2 s)')
        return flux / energy ** 2


class CrabSpectrum(object):
    """
    Crab spectral model.

    The following references are available:

        * 'meyer', 2010arXiv1008.4524M
        * 'hegra', 2004ApJ...614..897A
        * 'hess_pl', 2006A&A...457..899A
        * 'hess_ecpl', 2006A&A...457..899A

    Parameters
    ----------
    reference : {'meyer', 'hegra', 'hess_pl', 'hess_ecpl'}
        Which reference to use for the spectral model.
    """
    def __init__(self, reference='meyer'):
        if reference == 'meyer':
            model = MeyerCrabModel()
        elif reference == 'hegra':
            model = PowerLaw(**hegra)
        elif reference == 'hess_pl':
            model = PowerLaw(**hess_pl)
        elif reference == 'hess_ecpl':
            model = ExponentialCutoffPowerLaw(**hess_ecpl)
        else:
            raise ValueError("Unknown reference, choose one of the following:"
                             "'meyer', 'hegra', 'hess_pl' or 'hess_ecpl'")
        self.model = model
        self.reference = reference

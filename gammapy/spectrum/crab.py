# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from .models import PowerLaw, ExponentialCutoffPowerLaw, SpectralModel
from ..utils.modeling import ParameterList, Parameter

__all__ = [
    'CrabSpectrum',
]

# HESS publication: 2006A&A...457..899A
hess_pl = {'amplitude': 3.45e-11 * u.Unit('1 / (cm2 s TeV)'),
           'index': 2.63,
           'reference': 1 * u.TeV}

hess_ecpl = {'amplitude': 3.76e-11 * u.Unit('1 / (cm2 s TeV)'),
             'index': 2.39,
             'lambda_': 1 / (14.3 * u.TeV),
             'reference': 1 * u.TeV}

# HEGRA publication : 2004ApJ...614..897A
hegra = {'amplitude': 2.83e-11 * u.Unit('1 / (cm2 s TeV)'),
         'index': 2.62,
         'reference': 1 * u.TeV}


class MeyerCrabModel(SpectralModel):
    """Meyer 2010 log polynomial Crab spectral model.

    See 2010A%26A...523A...2M, Appendix D.
    """

    def __init__(self):
        coefficients = np.array([-0.00449161, 0, 0.0473174, -0.179475,
                                 -0.53616, -10.2708])
        self.parameters = ParameterList([
            Parameter('coefficients', coefficients)
        ])

    @staticmethod
    def evaluate(energy, coefficients):
        polynomial = np.poly1d(coefficients)
        log_energy = np.log10(energy.to('TeV').value)
        log_flux = polynomial(log_energy)
        flux = np.power(10, log_flux) * u.Unit('erg / (cm2 s)')
        return flux / energy ** 2


class CrabSpectrum(object):
    """Crab nebula spectral model.

    The Crab nebula is often used as a standard candle in gamma-ray astronomy.
    Fluxes and sensitivities are often quoted relative to the Crab spectrum.

    The following references are available:

    * 'meyer', http://adsabs.harvard.edu/abs/2010A%26A...523A...2M, Appendix D
    * 'hegra', http://adsabs.harvard.edu/abs/2000ApJ...539..317A
    * 'hess_pl' and 'hess_ecpl': http://adsabs.harvard.edu/abs/2006A%26A...457..899A

    Parameters
    ----------
    reference : {'meyer', 'hegra', 'hess_pl', 'hess_ecpl'}
        Which reference to use for the spectral model.

    Examples
    --------
    Let's first import what we need::

        import astropy.units as u
        from gammapy.spectrum import CrabSpectrum
        from gammapy.spectrum.models import PowerLaw

    Plot the 'hess_ecpl' reference Crab spectrum between 1 TeV and 100 TeV::

        crab_hess_ecpl = CrabSpectrum('hess_ecpl')
        crab_hess_ecpl.model.plot([1, 100] * u.TeV)

    Use a reference crab spectrum as unit to measure a differential flux (at 10 TeV)::

        >>> pwl = PowerLaw(index=2.3, amplitude=1e-12 * u.Unit('1 / (cm2 s TeV)'), reference=1 * u.TeV)
        >>> crab = CrabSpectrum('hess_pl').model
        >>> energy = 10 * u.TeV
        >>> dnde_cu = (pwl(energy) / crab(energy)).to('%')
        >>> print(dnde_cu)
        6.19699156377 %

    And the same for integral fluxes (between 1 and 10 TeV)::

        >>> # compute integral flux in crab units
        >>> emin, emax = [1, 10] * u.TeV
        >>> flux_int_cu = (pwl.integral(emin, emax) / crab.integral(emin, emax)).to('%')
        >>> print(flux_int_cu)
        3.5350582166 %
    """

    references = [
        'meyer', 'hegra', 'hess_pl', 'hess_ecpl',
    ]
    """Available references (see class docstring)."""

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
            fmt = 'Invalid reference: {!r}. Choices: {!r}'
            raise ValueError(fmt.format(reference, self.references))

        self.model = model
        self.reference = reference

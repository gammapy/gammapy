# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.units import Unit
from .models import TableModel

mbarn_to_cm2 = Unit('mbarn').to(Unit('cm^2'))

class Pion(object):
    """Pion decay gamma-ray SED calculator

    Compute gamma ray spectrum from pion decay
    for a given proton spectrum and target density.

    Reference:
    Kelner et al. (2006)
    http://adsabs.harvard.edu/abs/2006PhRvD..74c4018K
    """
    m_pi = 1.22e9  # pion mass (eV)
    m_p = 1e12  # proton mass (eV)
    # pion production energy threshold (eV)
    e_thresh = m_p + 2 * m_pi + m_pi ** 2 / (2 * m_p)

    def sigma(self, e_proton):
        """Inelastic part of the total p-p interaction cross section (cm^2)
        for a given proton energy (TeV)"""
        L = np.log(e_proton)
        sigma = ((34.3 + 1.88 * L + 0.25 * L ** 2) *
                 (1 - (self.e_thresh / e_proton) ** 4))
        return mbarn_to_cm2 * sigma

    def __init__(self, proton_spectrum, n_H=1):
        """
        proton_spectrum: proton spectrum (eV^-1)
        n_H: density (cm^-3)
        """
        self.proton_spectrum = proton_spectrum
        self.n_H = n_H

        self.L = np.log()

    def __call__(self, energies):
        """
        Returns: gamma spectrum (s^-1 eV^-1)
        energies: gamma energies (eV)
        """
        fluxes = np.zeros_like(energies)
        for e, f in zip(energies, fluxes):
            f = self.total_emissivity(e)

        return TableModel(energies, fluxes)

    def _integrand(self, e_gamma, e_proton):
        return (self.sigma(e_proton) *
                self.emissivity(e_gamma, e_proton) *
                self.proton_spectrum(e_proton) / e_proton)

    def emissivity(self, e_gamma, e_proton):
        """Emissivity for protons at one energy (both e_gamma and e_proton in TeV)"""

        # Expressions involving only the proton energy
        L = np.log(e_proton)
        B = 1.30 + 0.14 * L + 0.011 * L * L
        beta = 1. / (1.79 + 0.11 * L + 0.008 * L * L)
        k = 1. / (0.801 + 0.049 * L + 0.014 * L * L)

        # Expressions involving both proton and gamma energy
        x = e_gamma / e_proton
        logx = np.log(x)
        xbeta = x ** beta
        Delta = (1. / logx -
                 (4 * beta * xbeta) / (1 - xbeta) -
                 (4 * k * beta * xbeta * (1 - 2 * xbeta)) /
                 (1 + k * xbeta * (1 - xbeta)))

        F_gamma = (B * (logx / x) *
                   ((1 - xbeta) / (1 + k * xbeta * (1 - xbeta))) ** 4 *
                   Delta)

        return F_gamma

    def total_emissivity(self, e_gamma):
        """Emissivity for protons at all energies"""
        # flux = c * self.n_H * quad(self._integrand, )
        # return flux
        return NotImplementedError

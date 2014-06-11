# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.units import Unit
from .models import TableModel

__all__ = ['PionDecaySpectrum']

mbarn_to_cm2 = Unit('mbarn').to(Unit('cm^2'))


class PionDecaySpectrum(object):
    """Pion decay gamma-ray SED calculator.

    Compute gamma ray spectrum from pion decay
    for a given proton spectrum and target density.

    Reference:
    Kelner et al. (2006)
    http://adsabs.harvard.edu/abs/2006PhRvD..74c4018K

    Parameters
    ----------
    proton_spectrum : array_like
        proton spectrum (eV^-1)
    n_H : float
        density (cm^-3)

    See also
    --------
    gammapy.spectrum.InverseComptonSpectrum, gammafit.ProtonOZM
    """
    m_pi = 1.22e9  # pion mass (eV)
    m_p = 1e12  # proton mass (eV)
    # pion production energy threshold (eV)
    e_thresh = m_p + 2 * m_pi + m_pi ** 2 / (2 * m_p)

    def __init__(self, proton_spectrum, n_H=1):
        self.proton_spectrum = proton_spectrum
        self.n_H = n_H

        #self.L = np.log()

    def sigma(self, e_proton):
        """Inelastic part of the total p-p interaction cross section.

        Parameters
        ----------
        e_proton : array_like
            Proton energy (TeV)

        Returns
        -------
        cross_section : array
            Cross section (cm^2)
        """
        L = np.log(e_proton)
        sigma = ((34.3 + 1.88 * L + 0.25 * L ** 2) *
                 (1 - (self.e_thresh / e_proton) ** 4))
        return mbarn_to_cm2 * sigma

    def __call__(self, energies):
        """Compute spectrum.

        Parameters
        ----------
        energies : array_like
            Gamma energies (eV)

        Returns
        -------
        spectrum : TODO
            Gamma spectrum (s^-1 eV^-1)
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
        """Emissivity for protons at one energy.

        Parameters
        ----------
        e_gamma, e_proton : array_like
            Gamma-ray and proton energy (TeV)

        Returns
        -------
        TODO
        """

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
        """Emissivity for protons at all energies.

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        # flux = c * self.n_H * quad(self._integrand, )
        # return flux
        return NotImplementedError

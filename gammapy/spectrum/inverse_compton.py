# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy import log, sqrt, pi, log10
from astropy import constants as const
from .models import BlackBody, TableModel

__all__ = ['InverseCompton']

# Define some constants
c = const.c.cgs.value
h_eV = const.h.to('eV s').value
hbar = const.hbar.cgs.value
k_B = const.k_B.cgs.value
k_B_eV = const.k_B.to('eV/K').value
m_e_eV = (const.m_e * const.c ** 2).to('eV').value
sigma_T = 6.652458558e-25 # Thomson cross section


class InverseCompton(object):
    """Inverse comption (IC) SED calculator.
    
    Input: arbitrary photon and electron distribution
    Output: IC spectrum
    """

    def __init__(self, seed_ph_spec=BlackBody(T=2.7)):
        self.seed_ph_spec = seed_ph_spec

    def _Gamma_e(self, gamma, E_ini):
        """Gamma_e as defined in Blumenthal and Gould 1970"""
        return 4 * gamma * E_ini / m_e_eV

    def _f(self, q, Gamma_e):
        """General IC cross section."""
        return (2 * q * log(q) + (1 + 2 * q) * (1 - q) +
                0.5 * (1 - q) * (Gamma_e * q) ** 2 / (1 + Gamma_e * q))

    def _q(self, E_fin, gamma, Gamma_e):
        """q as defined in Blumenthal and Gould 1970 """
        return E_fin / (Gamma_e * (gamma * m_e_eV - E_fin))

    def loss_rate_per_energy(self, E_fin, E_ini, gamma, n_e):
        """Total compton spectrum as defined in Blumenthal and Gould 1970"""
        sigma_T = 6.652458558e-25 # Thomson cross section
        n_ph = self.seed_ph_spec(E_ini)

        return (3. / 4 * sigma_T * c * n_ph /
                E_ini * n_e(gamma) / gamma ** 2 *
                self._f(self._q(E_fin, gamma,
                          self._Gamma_e(gamma, E_ini)),
               self._Gamma_e(gamma, E_ini)))

    def _get_seed_ph_e(self, n_bins):
        E_ini_min = 1e-10
        E_ini_max = 1e3
        # Energy range of the initial photon distribution
        E_ini = np.logspace(log10(E_ini_min), log10(E_ini_max), n_bins)
        return E_ini

    def __call__(self, E_fin, n_e):
        """
        Perfoms integration over n_ph and n_e.
        """
        # Gamma integration range
        gamma_max = 1e13
        n_ph_bins = 200

        e_ph = self._get_seed_ph_e(n_ph_bins)
        de_ph = np.diff(e_ph)

        L_fin = np.empty_like(E_fin)
        for i, E_f in enumerate(E_fin):
            # Perform integration over E_ini (Initial photon distribution)
            L_gamma = np.array([])

            for E_i in e_ph[:-1]:
                # Determine gamma integration range
                gamma_min = np.array([sqrt(E_f / (4 * E_i) + (E_f / (2 * m_e_eV)) ** 2) +
                                      E_f / (2 * m_e_eV), 1.]).max()
                gamma = np.logspace(log10(gamma_min), log10(gamma_max), n_ph_bins)

                # Perform integration over gamma (Electron distribution)
                loss = self.loss_rate_per_energy(E_f, E_i, gamma[:-1], n_e)
                L_gamma = np.append(L_gamma, (np.diff(gamma) * loss).sum())

            L_fin[i] = np.sum(L_gamma * de_ph)

        return TableModel(E_fin, L_fin)

    def analytical(self, E_fin, index=1.5, T=2.7):
        """
        Analytical solution for a powerlaw electron and blackbody photon spectrum. Taken from Blumenthal and Gould 1970.
        """
        # P denotes the spectral index of the electron distribution
        # The values of F ar given in the paper
        F = {0: 3.48, 0.5: 3.00, 1.0: 3.20, 1.5: 3.91, 2.0: 5.25, 2.5: 7.57}
        return ((3 * pi * sigma_T) / (h_eV ** 3 * c ** 2) *
                (k_B_eV * T) ** ((index + 5) / 2.) * F[index] *
                E_fin ** (-(index + 1) / 2.))

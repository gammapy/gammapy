# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Supernova remnant (SNR) source models"""
from __future__ import division
import numpy as np
from numpy import log
from astropy.units import Unit
from ..utils.coordinates import flux_to_luminosity
from astropy import constants as const

__all__ = ['SNR', 'SNR_Truelove']

M_SUN = const.M_sun.cgs.value
M_P = const.m_p.cgs.value
K_B = const.k_B.cgs.value
SEC_TO_YEAR = Unit('second').to(Unit('year'))
YEAR_TO_SEC = 1. / SEC_TO_YEAR
CM_TO_PC = Unit('cm').to(Unit('pc'))

DEFAULT_N_ISM = 1
DEFAULT_E_SN = 1e51
DEFAULT_THETA = 0.1
DEFAULT_M = M_SUN

class SNR(object):
    """Supernova remnant (SNR) evolution models.

    Parameters
    ----------
    E_SN : float
        SNR energy (erg), equal to the SN energy after neutrino losses
    theta : float
        Fraction of E_SN that goes into cosmic rays
    n_ISM : float
        ISM density (g cm^-3)
    m : float
        Ejecta mass (g)
    T_stop : float
        Post-shock temperature where gamma-ray emission stops (K)
    """
    # Reference values for scaling formulas
    E_SN_ref = DEFAULT_E_SN
    theta_ref = DEFAULT_THETA
    n_ISM_ref = DEFAULT_N_ISM
    rho_ISM_ref = M_P * DEFAULT_N_ISM
    m_ref = DEFAULT_M
    T_stop_ref = 1e6

    def __init__(self, E_SN=DEFAULT_E_SN, theta=DEFAULT_THETA, n_ISM=DEFAULT_N_ISM,
                 m=DEFAULT_M, T_stop=1e6):
        self.E_SN = E_SN
        self.theta = theta
        self.n_ISM = n_ISM
        self.rho_ISM = n_ISM * M_P
        self.m = m
        self.T_stop = T_stop

        # Characteristic dimensions
        self.r_c = m ** (1. / 3) * self.rho_ISM ** (-1. / 3)
        self.t_c = E_SN ** (-1. / 2) * m ** (5. / 6) * self.rho_ISM ** (-1. / 3)

    def r_out(self, t):
        """Outer shell radius (pc) at age t (yr)"""
        # Timescale which determines the end of the free expansion (where r ~ t)
        # and the start of the sedov phase (where r ~ t^2/5)
        term1 = (self.E_SN / self.E_SN_ref) ** (-1. / 2)
        term2 = (self.m / self.m_ref) ** (5. / 6)
        term3 = (self.rho_ISM / self.rho_ISM_ref) ** (-1. / 3)
        tf = 200 * term1 * term2 * term3

        # proportional constant for the free expansion phase
        term1 = (self.E_SN / self.E_SN_ref) ** (1. / 2)
        term2 = (self.m / self.m_ref) ** (-1. / 2) 
        A = 0.01 * term1 * term2
        # The proportional constant for the sedov phase is choosen such,
        # that the function is continuous at tf.
        r1 = A * t
        r2 = A * tf ** (3. / 5) * t ** (2. / 5)
        return np.where(t < tf, r1, r2)

    def r_out_1(self, t):
        """Outer shell radius (pc) at age t (yr)"""
        # t in seconds
        t = YEAR_TO_SEC * t
        # Sedov Taylor Phase
        t_ST = 0.52 * self.t_c
        R_FE = self.r_free_expansion(t_ST)
        term1 = R_FE ** (5. / 2)
        term2 = (2.026 * (self.E_SN / self.rho_ISM)) ** (1. / 2)
        R_ST = (term1 + term2 * (t - t_ST)) ** (2. / 5)
        return CM_TO_PC * np.select([t < t_ST, t >= t_ST], [R_FE, R_ST])

    def r_in(self, t):
        """Inner shell radius (pc) at age t (yr)"""
        return self.r_out(t) * (1 - 0.0914)

    def r_free_expansion(self, t):
        """Reverse shock radius (pc) at age t (yr)
        during free expansion phase"""
        return 1.12 * self.r_c * (t / self.t_c) ** (2. / 3)

    def r_reverse(self, t):
        """Reverse shock radius (pc) at age t (yr)

        Reference: Gelfand & Slane 2009, Appendix A.
        """
        # t in seconds
        t = YEAR_TO_SEC * t
        # Time when reverse shock reaches the "core"
        t_core = 0.25 * self.t_c
        
        term1 = (t - t_core) / (self.t_c)
        term2 = (1.49 - 0.16 * term1 - 0.46 * log(t / t_core))
        R_1 = self.r_free_expansion(t) / 1.19
        R_RS = term2 * (self.r_c / self.t_c) * t
        return CM_TO_PC * np.select([t < t_core, t >= t_core], [R_1, R_RS])

    def L(self, t):
        """Gamma-ray luminosity above 1 TeV (ph s^-1) at age t (yr).

        The luminosity is assumed constant in a given age
        interval and zero before and after.
        Reference: Axel bachelor thesis.

        It is computed using the following formula:
        Reference: Drury, Aharonian, Voelk 1994 Equation 9.
        """
        t_start = self._t_start()
        t_end = self._t_end()

        # Flux in 1 kpc distance according to Drury formula 9
        term1 = self.E_SN / self.E_SN_ref
        term2 = self.n_ISM / self.n_ISM_ref
        F = 9e-11 * self.theta * term1 * term2

        # Corresponding luminosity
        L = flux_to_luminosity(F, distance=1)

        # Apply time interval
        L = np.select([t <= t_start, t <= t_end], [0, L])
        return L

    def _t_start(self):
        term1 = 200 * (self.E_SN / self.E_SN_ref) ** (-1. / 2)
        term2 = (self.m / self.m_ref) ** (5. / 6)
        term3 = (self.rho_ISM / self.rho_ISM_ref) ** (-1. / 3)         
        return term1 * term2 * term3

    def _t_end(self):
        term1 = 3 * M_P / (100 * K_B * self.T_stop)
        term2 = (self.E_SN / self.rho_ISM) ** (2. / 5)
        return SEC_TO_YEAR * (term1 * term2) ** (5. / 6)


class SNR_Truelove(SNR):
    """SNR model according to Truelove & McKee (1999).

    Reference: Gelfand & Slane 2009, Appendix A.
    """

    def r_out(self, t):
        """Outer shell radius (pc) at age t (yr).
        """
        t = YEAR_TO_SEC * t
        # Sedov Taylor Phase
        t_ST = 0.52 * self.t_c
        R_FE = self.r_reverse_free_expansion(t_ST)
        term1 = R_FE ** (5. / 2)
        term2 = (2.026 * (self.E_SN / self.rho_ISM)) ** (1. / 2)
        R_ST = (term1 + term2 * (t - t_ST)) ** (2. / 5)
        return CM_TO_PC * np.select([t < t_ST, t >= t_ST], [R_FE, R_ST])

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pulsar wind nebula (PWN) source models"""
from __future__ import division
import logging
import numpy as np
from astropy.units import Unit
from . import snr, pulsar

__all__ = ['PWN']

YEAR_TO_SEC = Unit('year').to(Unit('second'))
CM_TO_PC = Unit('cm').to(Unit('pc'))
PC_TO_CM = 1. / CM_TO_PC

class PWN(object):
    """Pulsar wind nebula (PWN) evolution model.

    TODO: document

    Parameters
    ----------
    eta_e : float
        Fraction of energy going into electrons.
    eta_B : float
        Fraction of energy going into magnetic fields.
    """
    def __init__(self, pulsar=pulsar.ModelPulsar(), snr=snr.SNR(),
                 eta_e=0.999,
                 eta_B=0.001,
                 q_type='constant',
                 r_type='constant',
                 B_type='constant'):
        self.pulsar = pulsar
        self.snr = snr
        self.eta_e = eta_e
        self.eta_B = eta_B
        # Spectrum is not needed for now
        # self.electron_spec = Spectrum.table_spectrum(emin, emax, npoints)
        # Choose appropriate evolution functions for the
        # requested model
        # q = {'q':'q'}
        # Bs = {'constant':{'function': self.B_constant,
        #                  'params': B},
        #      'spindown':{'function': self.B_spindown,
        #                  'params': [L0, tau0, n, eta_B]}}
        # qs = ['q_constant', 'q_spindown', 'q_burst']
        #
        # self.B = Bs[B_type]['function'](Bs[B_type]['params'])
        # self.B = self.B_constant(B=10)

    def r_free(self, t):
        """Radius (pc) at age t (yr) during free expansion phase.

        Reference: Gaensler & Slane 2006 (Formula 8).
        """
        term1 = (self.snr.E_SN ** 3 * self.pulsar.L_0 ** 2) / (self.snr.m ** 5)
        term2 = (t * YEAR_TO_SEC) ** (6. / 5)
        return CM_TO_PC * 1.44 * term1 ** (1. / 10) * term2

    def r(self, t):
        """Radius (pc) at age t (yr).

        After collision with the reverse shock we assume constant radius.

        TODO: shouldn't we assume constant fraction of snr.r_out ???
        """
        from scipy.optimize import fsolve
        t = YEAR_TO_SEC * t
        t_coll = fsolve(lambda x: self.r_free(x) - self.snr.r_reverse(x), 4e3)
        # 4e3 years is a typical value that works for fsolve
        # Radius at time of collision
        r_coll = self.r_free(t_coll)
        return np.where(t < t_coll, self.r_free(t), r_coll)

    def B(self, t):
        """Evolution of the magnetic field inside the PWN.
        
        Eta denotes the fraction of the spin down energy
        converted to magnetic field energy.

        TODO: Check this formula and give reference!
        """
        return np.sqrt((3 * self.eta_B * self.snr.L0 *
                     YEAR_TO_SEC * self.pulsar.tau_0) /
                   (self.r(t) * PC_TO_CM) ** 3 *
                   (1 - (1 + (t / self.pulsar.tau_0)) ** (-1)))

    def L(self, t):
        """Simple luminosity evolution model.
        
        Assumes that the luminosity just follows the total energy content.
        """
        return (3e-16 * self.pulsar.tau_0 * self.pulsar.L_0 *
                (t / (t + self.pulsar.tau_0)))

    def B_constant(self, B):
        def B_constant(e, t):
            return B
        return B_constant

    def B_spindown(self, L0, tau0, n, eta_B):
        def B_spindown(e, t):
            return self.L(t) / self.eta_B
        return B_spindown

    def L_constant(self, L0):
        def f(t):
            return L0
        return f

    def L_spindown(self, L0, tau0, n=3):
        """Pulsar spin-down luminosity (erg s^-1).
        
        Parameters
        ----------
        L0 : float
            Luminosity at time tau0 (erg s^-1)
        tau0 : float
            Spin-down timescale (yr)
        n : flaot
            Braking index
        """
        beta = -(n + 1) / (n - 1)

        def f(t):
            return L0 * (1 + t / tau0) ** beta
        return f

    def q(self, e, t, norm=1, e0=1, index=-2, emin=1e1, emax=1e4, burst=False):
        """Injection spectrum: exponential cutoff power law.
        
        Parameters
        ----------
        TODO
        """
        # For burst-like injection only inject at the first time step
        if burst and t != 0:
            return np.zeros_like(e)
        # q = power_law(e, norm, e0, index)
        # q[(e < emin) | (e > emax)] = 0
        # return q
        raise NotImplementedError

    def energy_loss_rate(self, B=10, w_rad=0.25):
        """Energy losses: synchrotron, IC, adiabatic.
        
        Parameters
        ----------
        TODO
        """
        # Synchrotron and IC losses:
        b = 1e-5

        def p(e, t):
            """Energy loss rate (TeV s^-1) at a given
            energy (TeV) and time (s)"""
            return b * e ** 2
        return p

    def evolve(self, age=1e3, dt=1):
        """Evolve the electron spectrum in time.
        
        From the current age to the new requested age.

        If the current age is larger than the requested age,
        the PWN is reset to age 0 and then evolved to the requested age.

        *Method*
        
        
        The evolution is described by the continuity equation::
            $$dn / dt = d(pn) / de + q$$

        e: energy array (TeV)
        n: electron spectrum array (TeV^-1)
        p = p(e, t): energy loss function
        q = q(e, t): injection function

        *Implementation*

        We are using a simple finite difference method to solve the
        partial differential equation.

        Spectra can have sharp cutoffs, i.e. infinite derivative wrt. de.
        This will numerically blow up quickly to +- inf.
        Even worse, this instability will propagate one energy bin to the
        left in each time step, if the difference (right - center) is used.
        If (center - left) is used, it will propagate to the right.

        To avoid this problem, we compute both the left and the right difference
        and then take the smaller absolute value::

            dpn_right[i] = pn[i+1] - pn[i]
            dpn_left[i] = pn[i] - pn[i-1]
            dpn = where( abs(dpn_right) < abs(dpn_left), dpn_right, dpn_left)

        This avoids the numerical instability as long as the spectrum is smooth
        on one of the two sides for each energy.
        E.g. a power-law spectrum that is 0 outside a range [emin, emax] is not
        a problem any more.

        Parameters
        ----------
        age : float
            Final age (yr)
        dt : float
            Time step (yr)
        """
        if self.age > age:
            logging.info('current age = {0} > requested age = {1}'
                         ''.format(self.age, age))
            logging.info('Resetting to age 0.')
            self.age = 0
            self.electron_spec.y = 0
            self.evolve(age, dt)
        # Make sure we evolve exactly to the requested age
        nsteps = int((age - self.age) / dt)
        dt = (age - self.age) / nsteps
        logging.info('current age = {0}, requested age = {1},'
                     ' dt = {2}, nsteps = {3}'
                     ''.format(self.age, age, dt, nsteps))
        # Convenient shortcuts for current electron spectrum
        # Note that these are references, no copying takes place.
        e = self.electron_spec.e
        n = self.electron_spec.n
        de = self._get_diff(e)
        logging.info('Starting evolution')
        for t in np.linspace(self.age, age, nsteps):
            pn = n * self.p(e, t)
            # See docstring for an explanation why we implement
            # the difference in this way:
            dpn_right = self._get_diff(pn, side='right')
            dpn_left = self._get_diff(pn, side='left')
            dpn = np.where(abs(dpn_right) < abs(dpn_left), dpn_right, dpn_left)
            n += dt * (dpn / de + self.q(e, t))
        logging.info('Evolution finished')

    def _get_diff(self, e, side='right'):
        """Implement diff for an array, including handling of end points.
        
        side = 'right': diff[i] = e[i + 1] - e[i]
        side = 'left': diff[i] = e[i] - e[i - 1]
        """
        e = e if side == 'right' else e[::-1]
        return np.ediff1d(e, to_end=e[-1] - e[-2])

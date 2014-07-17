# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pulsar source models"""
from __future__ import print_function, division
from numpy import pi, sqrt
from astropy.units import Unit, Quantity

__all__ = ['Pulsar', 'ModelPulsar']

SEC_TO_YEAR = Unit('second').to(Unit('year'))
YEAR_TO_SEC = 1. / SEC_TO_YEAR

DEFAULT_I = Quantity(1e45, 'g cm^2').value  # moment of inertia (g cm^2)
DEFAULT_R = Quantity(1e6, 'cm').value   # radius (cm)
B_CONST = Quantity(3.2e19, 'gauss').value   # TODO: document


class Pulsar(object):
    """Observed pulsar with known period and period derivative.

    Reference: TODO

    Parameters
    ----------
    P : float
        Rotation period (sec)
    Pdot : float
        Rotation period derivative (sec sec^-1)
    I : float
        Moment of inertia (g cm^2)
    R : float
        Radius (cm)
    """

    def __init__(self, P, Pdot, I=DEFAULT_I, R=DEFAULT_R):
        self.P = P
        self.Pdot = Pdot
        self.I = I
        self.R = R

    @property
    def L(self):
        """Spin-down luminosity (erg s^-1)."""
        return 4 * pi ** 2 * self.I * self.Pdot / self.P ** 3

    @property
    def tau(self):
        """Characteristic age (yr)."""
        return SEC_TO_YEAR * self.P / (2 * self.Pdot)

    @property
    def B(self):
        """Magnetic field strength at the polar cap (Gauss)."""
        return B_CONST * sqrt(self.P * self.Pdot)


class ModelPulsar(Pulsar):
    """Model pulsar with known parameters and magnetic dipole spin-down.

    Reference: TODO

    Parameters
    ----------
    P_0 : float
        Period at birth (erg s^-1)
    logB : float
        Logarithm of the magnetic field, which is constant (log Gauss)
    n : float
        Spin-down braking index
    I : float
        Moment of inertia (g cm^2)
    R : float
        Radius (cm)
    """

    def __init__(self, P_0=0.1, logB=10, n=3, I=DEFAULT_I, R=DEFAULT_R):
        self.I = I
        self.R = R
        self.P_0 = P_0
        self.logB = logB
        self.Pdot_0 = (10 ** self.logB) ** 2 / (P_0 * B_CONST ** 2)
        self.L_0 = 4 * pi ** 2 * self.I * (self.Pdot_0 / P_0 ** 3)
        self.tau_0 = SEC_TO_YEAR * P_0 / (2 * self.Pdot_0)
        self.n = float(n)
        self.beta = (n + 1.) / (n - 1.)

    def L(self, t):
        """Spin down luminosity (erg sec^-1) at age t (yr)."""
        return self.L_0 * (1 + (t / self.tau_0)) ** self.beta

    def P(self, t):
        """Period (sec) at age t (yr)."""
        return self.P_0 * (1 + (t / self.tau_0)) ** self.beta

    def E(self, t):
        """Total released energy (erg) at age t (yr).

        This is simply the time-integrated spin-down luminosity since birth.
        """
        return self.L_0 * YEAR_TO_SEC * self.tau_0 * (t / (t + self.tau_0))

    def Pdot(self, t):
        """Period derivative (sec sec^-1) at age t (yr).

        Pdot[s/s] for a given period P[s] and magnetic field B[log10 Gauss],
        assuming a dipole spin-down.
        """
        return (10 ** self.logB) ** 2 / (self.P(t) * B_CONST ** 2)

    def CharAge(self, t):
        """Characteristic age (yr) at real age t (yr)."""
        return SEC_TO_YEAR * self.P(t) / (2 * self.Pdot(t))

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pulsar source models."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from ...extern.validator import validate_physical_type

__all__ = ["Pulsar", "SimplePulsar"]

DEFAULT_I = Quantity(1e45, "g cm2")
"""Pulsar default moment of inertia"""

DEFAULT_R = Quantity(1e6, "cm")
"""Pulsar default radius of the neutron star"""

B_CONST = Quantity(3.2e19, "gauss s^(-1/2)")
"""Pulsar default magnetic field constant"""


class SimplePulsar(object):
    """Magnetic dipole spin-down model for a pulsar.

    Reference: http://www.cv.nrao.edu/course/astr534/Pulsars.html

    Parameters
    ----------
    P : `~astropy.units.Quantity`
        Rotation period (sec)
    P_dot : `~astropy.units.Quantity`
        Rotation period derivative (sec sec^-1)
    I : `~astropy.units.Quantity`
        Moment of inertia (g cm^2)
    R : `~astropy.units.Quantity`
        Radius of the pulsar (cm)
    """

    def __init__(self, P, P_dot, I=DEFAULT_I, R=DEFAULT_R):
        validate_physical_type("P", P, "time")
        validate_physical_type("P_dot", P_dot, "dimensionless")
        self.P = P
        self.P_dot = P_dot
        self.I = I
        self.R = R

    @property
    def luminosity_spindown(self):
        """Spin-down luminosity (`~astropy.units.Quantity`).

        Notes
        -----
        The spin-down luminosity is given by:

        .. math::

            \\dot{L} = 4\\pi^2 I \\frac{\\dot{P}}{P^{3}}
        """
        return 4 * np.pi ** 2 * self.I * self.P_dot / self.P ** 3

    @property
    def tau(self):
        """Characteristic age (`~astropy.units.Quantity`).

        Notes
        -----
        The characteristic age is given by:

        .. math::

            \\tau = \\frac{P}{2\\dot{P}}
        """
        return (self.P / (2 * self.P_dot)).to("yr")

    @property
    def magnetic_field(self):
        """Magnetic field strength at the polar cap (`~astropy.units.Quantity`).

        Notes
        -----
        The magnetic field is given by:

        .. math::

            B = 3.2\\cdot 10^{19} (P\\dot{P})^{1/2} [\\textnormal(Gauss)]
        """
        return B_CONST * np.sqrt(self.P * self.P_dot)


class Pulsar(SimplePulsar):
    """Magnetic dipole spin-down pulsar model.

    Reference: http://www.cv.nrao.edu/course/astr534/Pulsars.html

    Parameters
    ----------
    P_0 : float
        Period at birth
    logB : float
        Logarithm of the magnetic field, which is constant
    n : float
        Spin-down braking index
    I : float
        Moment of inertia
    R : float
        Radius
    """

    def __init__(
        self,
        P_0=Quantity(0.1, "s"),
        logB=10,
        n=3,
        I=DEFAULT_I,
        R=DEFAULT_R,
        age=None,
        L_0=None,
        morphology="Delta2D",
    ):
        self.I = I
        self.R = R
        self.P_0 = P_0
        self.logB = logB
        self.P_dot_0 = (Quantity(10 ** logB, "gauss") / B_CONST) ** 2 / P_0
        self.tau_0 = P_0 / (2 * self.P_dot_0)
        self.n = float(n)
        self.beta = (n + 1.) / (n - 1.)
        self.morphology = morphology
        if age is not None:
            validate_physical_type("age", age, "time")
            self.age = age
        if L_0 is None:
            self.L_0 = 4 * np.pi ** 2 * self.I * self.P_dot_0 / self.P_0 ** 3

    def luminosity_tev(self, t=None, fraction=0.1):
        """Gamma-ray luminosity assumed to be a certain fraction of the spin-down luminosity.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.
        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")
        return self.luminosity_spindown(t) * fraction

    def luminosity_spindown(self, t=None):
        """Spin down luminosity  at age t.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.

        Notes
        -----
        The spin-down luminosity is given by:

        .. math::

            \\dot{L}(t) = \\dot{L}_0 \\left(1 + \\frac{t}{\\tau_0}\\right)^{\\frac{n + 1}{n - 1}}
        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")
        return self.L_0 * (1 + (t / self.tau_0)) ** self.beta

    def period(self, t=None):
        """Period at age t.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.

        Notes
        -----
        The period is given by:

        .. math::

            P(t) = P_0\\left(1 + \\frac{t}{\\tau_0}\\right)^{\\frac{1}{n - 1}}
        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")
        return self.P_0 * (1 + (t / self.tau_0)) ** self.beta

    def energy_integrated(self, t=None):
        """Total released energy at age t.

        Time-integrated spin-down luminosity since birth.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.

        Notes
        -----
        The time integrated energy is given by:

        .. math::

            E(t) = \\dot{L}_0 \\tau_0 \\frac{t}{t + \\tau_0}

        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")
        return self.L_0 * self.tau_0 * (t / (t + self.tau_0))

    def period_dot(self, t=None):
        """Period derivative at age t.

        P_dot for a given period and magnetic field B, assuming a dipole
        spin-down.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.

        Notes
        -----
        The period derivative is given by:

        .. math::

            \\dot{P}(t) = \\frac{B^2}{3.2 \\cdot 10^{19} P(t)}
        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")
        return Quantity(10 ** self.logB, "gauss") ** 2 / (self.period(t) * B_CONST ** 2)

    def tau(self, t=None):
        """Characteristic age at real age t.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.

        Notes
        -----
        The characteristic age is given by:

        .. math::

            \\tau = \\frac{P}{2\\dot{P}}

        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")
        return self.period(t) / 2 * self.period_dot(t)

    def magnetic_field(self, t=None):
        """Magnetic field strength at the polar cap. Assumed to be constant.

        Notes
        -----
        The magnetic field is given by:

        .. math::

            B = 3.2\\cdot 10^{19} (P\\dot{P})^{1/2} [\\textnormal(Gauss)]
        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")
        return B_CONST * np.sqrt(self.period(t) * self.period_dot(t))

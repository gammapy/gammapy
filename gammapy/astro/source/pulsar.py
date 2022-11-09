# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pulsar source models."""
import numpy as np
from astropy.units import Quantity

__all__ = ["Pulsar", "SimplePulsar"]

DEFAULT_I = Quantity(1e45, "g cm2")
"""Pulsar default moment of inertia"""

DEFAULT_R = Quantity(1e6, "cm")
"""Pulsar default radius of the neutron star"""

B_CONST = Quantity(3.2e19, "gauss s^(-1/2)")
"""Pulsar default magnetic field constant"""


class SimplePulsar:
    """Magnetic dipole spin-down model for a pulsar.

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

    def __init__(self, P, P_dot, I=DEFAULT_I, R=DEFAULT_R):  # noqa: E741
        self.P = Quantity(P, "s")
        self.P_dot = P_dot
        self.I = I  # noqa: E741
        self.R = R

    @property
    def luminosity_spindown(self):
        r"""Spin-down luminosity (`~astropy.units.Quantity`).

        .. math:: \dot{L} = 4\pi^2 I \frac{\dot{P}}{P^{3}}
        """
        return 4 * np.pi**2 * self.I * self.P_dot / self.P**3

    @property
    def tau(self):
        r"""Characteristic age (`~astropy.units.Quantity`).

        .. math:: \tau = \frac{P}{2\dot{P}}
        """
        return (self.P / (2 * self.P_dot)).to("yr")

    @property
    def magnetic_field(self):
        r"""Magnetic field strength at the polar cap (`~astropy.units.Quantity`).

        .. math:: B = 3.2 \cdot 10^{19} (P\dot{P})^{1/2} \text{ Gauss}
        """
        return B_CONST * np.sqrt(self.P * self.P_dot)


class Pulsar:
    """Magnetic dipole spin-down pulsar model.

    Parameters
    ----------
    P_0 : float
        Period at birth
    B : `~astropy.units.Quantity`
        Magnetic field strength at the poles (Gauss)
    n : float
        Spin-down braking index
    I : float
        Moment of inertia
    R : float
        Radius
    """

    def __init__(
        self,
        P_0="0.1 s",
        B="1e10 G",
        n=3,
        I=DEFAULT_I,  # noqa: E741
        R=DEFAULT_R,
        age=None,
        L_0=None,  # noqa: E741
    ):
        P_0 = Quantity(P_0, "s")
        B = Quantity(B, "G")

        self.I = I  # noqa: E741
        self.R = R
        self.P_0 = P_0
        self.B = B
        self.P_dot_0 = (B / B_CONST) ** 2 / P_0
        self.tau_0 = P_0 / (2 * self.P_dot_0)
        self.n = float(n)
        self.beta = -(n + 1.0) / (n - 1.0)
        if age is not None:
            self.age = Quantity(age, "yr")
        if L_0 is None:
            self.L_0 = 4 * np.pi**2 * self.I * self.P_dot_0 / self.P_0**3

    def luminosity_spindown(self, t):
        r"""Spin down luminosity.

        .. math::
            \dot{L}(t) = \dot{L}_0 \left(1 + \frac{t}{\tau_0}\right)^{-\frac{n + 1}{n - 1}}

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar
        """
        t = Quantity(t, "yr")
        return self.L_0 * (1 + (t / self.tau_0)) ** self.beta

    def energy_integrated(self, t):
        r"""Total energy released by a given time.

        Time-integrated spin-down luminosity since birth.

        .. math:: E(t) = \dot{L}_0 \tau_0 \frac{t}{t + \tau_0}

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.
        """
        t = Quantity(t, "yr")
        return self.L_0 * self.tau_0 * (t / (t + self.tau_0))

    def period(self, t):
        r"""Rotation period.

        .. math::
            P(t) = P_0 \left(1 + \frac{t}{\tau_0}\right)^{\frac{1}{n - 1}}

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar
        """
        t = Quantity(t, "yr")
        return self.P_0 * (1 + (t / self.tau_0)) ** (1.0 / (self.n - 1))

    def period_dot(self, t):
        r"""Period derivative at age t.

        P_dot for a given period and magnetic field B, assuming a dipole
        spin-down.

        .. math:: \dot{P}(t) = \frac{B^2}{3.2 \cdot 10^{19} P(t)}

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.
        """
        t = Quantity(t, "yr")
        return self.B**2 / (self.period(t) * B_CONST**2)

    def tau(self, t):
        r"""Characteristic age at real age t.

        .. math:: \tau = \frac{P}{2\dot{P}}

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.
        """
        t = Quantity(t, "yr")
        return self.period(t) / 2 * self.period_dot(t)

    def magnetic_field(self, t):
        r"""Magnetic field at polar cap (assumed constant).

        .. math::
            B = 3.2 \cdot 10^{19} (P\dot{P})^{1/2} \text{ Gauss}

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the pulsar.
        """
        t = Quantity(t, "yr")
        return B_CONST * np.sqrt(self.period(t) * self.period_dot(t))

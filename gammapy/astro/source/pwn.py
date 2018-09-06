# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pulsar wind nebula (PWN) source models."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from astropy.utils import lazyproperty
import astropy.constants as const
from ...extern.validator import validate_physical_type
from ..source import Pulsar, SNRTrueloveMcKee

__all__ = ["PWN"]


class PWN(object):
    """Simple pulsar wind nebula (PWN) evolution model.

    Parameters
    ----------
    pulsar : `~gammapy.astro.source.Pulsar`
        Pulsar model instance.
    snr : `~gammapy.astro.source.SNRTrueloveMcKee`
        SNR model instance
    eta_e : float
        Fraction of energy going into electrons.
    eta_B : float
        Fraction of energy going into magnetic fields.
    age : `~astropy.units.Quantity`
        Age of the PWN.
    morphology : str
        Morphology model of the PWN
    """

    def __init__(
        self,
        pulsar=Pulsar(),
        snr=SNRTrueloveMcKee(),
        eta_e=0.999,
        eta_B=0.001,
        morphology="Gaussian2D",
        age=None,
    ):
        self.pulsar = pulsar
        if not isinstance(snr, SNRTrueloveMcKee):
            raise ValueError("SNR must be instance of SNRTrueloveMcKee")
        self.snr = snr
        self.eta_e = eta_e
        self.eta_B = eta_B
        self.morphology = morphology
        if age is not None:
            validate_physical_type("age", age, "time")
            self.age = age

    def _radius_free_expansion(self, t):
        """Radius at age t during free expansion phase.

        Reference: http://adsabs.harvard.edu/abs/2006ARA%26A..44...17G (Formula 8).
        """
        term1 = (self.snr.e_sn ** 3 * self.pulsar.L_0 ** 2) / (self.snr.m_ejecta ** 5)
        return (1.44 * term1 ** (1. / 10) * t ** (6. / 5)).cgs

    @lazyproperty
    def _collision_time(self):
        """Time of collision between the PWN and the reverse shock of the SNR.

        Returns
        -------
        t_coll : `~astropy.units.Quantity`
            Time of collision.
        """
        from scipy.optimize import fsolve

        def time_coll(t):
            t = Quantity(t, "yr")
            r_pwn = self._radius_free_expansion(t).to("cm").value
            r_shock = self.snr.radius_reverse_shock(t).to("cm").value
            return r_pwn - r_shock

        # 4e3 years is a typical value that works for fsolve
        return Quantity(fsolve(time_coll, 4e3), "yr")

    def radius(self, t=None):
        """Radius of the PWN at age t.

        Reference: http://adsabs.harvard.edu/abs/2006ARA%26A..44...17G (Formula 8).

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.

        Notes
        -----
        During the free expansion phase the radius of the PWN evolves like:

        .. math::

            R_{PWN}(t) = 1.44\\text{pc}\\left(\\frac{E_{SN}^3\\dot{E}_0^2}
            {M_{ej}^5}\\right)^{1/10}t^{6/5}

        After the collision with the reverse shock of the SNR, the radius is
        assumed to be constant (See `~gammapy.astro.source.SNRTrueloveMcKee.radius_reverse_shock`)

        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")
        # Radius at time of collision
        r_coll = self._radius_free_expansion(self._collision_time)
        r = np.where(
            t < self._collision_time, self._radius_free_expansion(t).value, r_coll.value
        )
        return Quantity(r, "cm")

    def magnetic_field(self, t=None):
        """Estimate of the magnetic field inside the PWN.

        By assuming that a certain fraction of the spin down energy is
        converted to magnetic field energy an estimation of the magnetic
        field can be derived.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.
        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")

        energy = self.pulsar.energy_integrated(t)
        volume = 4. / 3 * np.pi * self.radius(t) ** 3
        return np.sqrt(2 * const.mu0 * self.eta_B * energy / volume)

    def luminosity_tev(self, t=None, fraction=0.1):
        """TeV luminosity from a simple evolution model.

        Assumes that the luminosity is just a fraction of the total energy content
        of the pulsar. No cooling is considered and therefore the estimate is very bad.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.
        """
        return fraction * self.pulsar.energy_integrated(t)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pulsar wind nebula (PWN) source models."""
import numpy as np
import scipy.optimize
import astropy.constants
from astropy.units import Quantity
from astropy.utils import lazyproperty
from .pulsar import Pulsar
from .snr import SNRTrueloveMcKee

__all__ = ["PWN"]


class PWN:
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
            self.age = Quantity(age, "yr")

    def _radius_free_expansion(self, t):
        """Radius at age t during free expansion phase.

        Reference: https://ui.adsabs.harvard.edu/abs/2006ARA%26A..44...17G (Formula 8).
        """
        term1 = (self.snr.e_sn**3 * self.pulsar.L_0**2) / (self.snr.m_ejecta**5)
        return (1.44 * term1 ** (1.0 / 10) * t ** (6.0 / 5)).cgs

    @lazyproperty
    def _collision_time(self):
        """Time of collision between the PWN and the reverse shock of the SNR.

        Returns
        -------
        t_coll : `~astropy.units.Quantity`
            Time of collision.
        """

        def time_coll(t):
            t = Quantity(t, "yr")
            r_pwn = self._radius_free_expansion(t).to_value("cm")
            r_shock = self.snr.radius_reverse_shock(t).to_value("cm")
            return r_pwn - r_shock

        # 4e3 years is a typical value that works for fsolve
        return Quantity(scipy.optimize.fsolve(time_coll, 4e3), "yr")

    def radius(self, t):
        r"""Radius of the PWN at age t.

        During the free expansion phase the radius of the PWN evolves like:

        .. math::
            R_{PWN}(t) = 1.44 \left(\frac{E_{SN}^3\dot{E}_0^2}
            {M_{ej}^5}\right)^{1/10}t^{6/5}
            \text{pc}

        After the collision with the reverse shock of the SNR, the radius is
        assumed to be constant (See `~gammapy.astro.source.SNRTrueloveMcKee.radius_reverse_shock`).

        Reference: https://ui.adsabs.harvard.edu/abs/2006ARA%26A..44...17G (Formula 8).

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        t = Quantity(t, "yr")
        r_collision = self._radius_free_expansion(self._collision_time)
        r = np.where(
            t < self._collision_time,
            self._radius_free_expansion(t).value,
            r_collision.value,
        )
        return Quantity(r, "cm")

    def magnetic_field(self, t):
        """Estimate of the magnetic field inside the PWN.

        By assuming that a certain fraction of the spin down energy is
        converted to magnetic field energy an estimation of the magnetic
        field can be derived.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        t = Quantity(t, "yr")
        energy = self.pulsar.energy_integrated(t)
        volume = 4.0 / 3 * np.pi * self.radius(t) ** 3
        return np.sqrt(2 * astropy.constants.mu0 * self.eta_B * energy / volume)

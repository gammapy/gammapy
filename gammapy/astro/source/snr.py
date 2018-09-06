# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Supernova remnant (SNR) source models."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
import astropy.constants as const
from astropy.utils import lazyproperty
from ...extern.validator import validate_physical_type

__all__ = ["SNR", "SNRTrueloveMcKee"]


class SNR(object):
    """Simple supernova remnant (SNR) evolution model.

    The model is based on the Sedov-Taylor solution for strong explosions.

    Reference: http://adsabs.harvard.edu/abs/1950RSPSA.201..159T

    Parameters
    ----------
    e_sn : `~astropy.units.Quantity`
        SNR energy (erg), equal to the SN energy after neutrino losses
    theta : `~astropy.units.Quantity`
        Fraction of E_SN that goes into cosmic rays
    n_ISM : `~astropy.units.Quantity`
        ISM density (g cm^-3)
    m_ejecta : `~astropy.units.Quantity`
        Ejecta mass (g)
    t_stop : `~astropy.units.Quantity`
        Post-shock temperature where gamma-ray emission stops.
    """

    def __init__(
        self,
        e_sn=Quantity(1e51, "erg"),
        theta=Quantity(0.1),
        n_ISM=Quantity(1, "cm-3"),
        m_ejecta=const.M_sun,
        t_stop=Quantity(1e6, "K"),
        age=None,
        morphology="Shell2D",
        spectral_index=2.1,
    ):
        self.e_sn = e_sn
        self.theta = theta
        self.rho_ISM = n_ISM * const.m_p
        self.n_ISM = n_ISM
        self.m_ejecta = m_ejecta
        self.t_stop = t_stop
        self.morphology = morphology
        self.spectral_index = spectral_index
        if age is not None:
            validate_physical_type("age", age, "time")
            self.age = age

    def radius(self, t=None):
        """Outer shell radius at age t.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.

        Notes
        -----
        The radius during the free expansion phase is given by:

        .. math::

            r_{SNR}(t) \\approx 0.01 \\textnormal{}
            \\left(\\frac{E_{SN}}{10^{51}erg}\\right)^{1/2}
            \\left(\\frac{M_{ej}}{M_{\\odot}}\\right)^{-1/2} t

        The radius during the Sedov-Taylor phase evolves like:

        .. math::

            r_{SNR}(t) \\approx \\left(\\frac{E_{SN}}{\\rho_{ISM}}\\right)^{1/5}t^{2/5}

        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")
        r = np.where(
            t > self.sedov_taylor_begin,
            self._radius_sedov_taylor(t).to("cm").value,
            self._radius_free_expansion(t).to("cm").value,
        )
        return Quantity(r, "cm")

    def _radius_free_expansion(self, t):
        """Shock radius at age t during free expansion phase.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.

        """
        # proportional constant for the free expansion phase
        term_1 = (self.e_sn / Quantity(1e51, "erg")) ** (1. / 2)
        term_2 = (self.m_ejecta / const.M_sun) ** (-1. / 2)
        return Quantity(0.01, "pc/yr") * term_1 * term_2 * t

    def _radius_sedov_taylor(self, t):
        """Shock radius  at age t  during Sedov Taylor phase.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.

        """
        R_FE = self._radius_free_expansion(self.sedov_taylor_begin)
        return R_FE * (t / self.sedov_taylor_begin) ** (2. / 5)

    def radius_inner(self, t, fraction=0.0914):
        """Inner radius  at age t  of the SNR shell.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.

        """
        return self.radius(t) * (1 - fraction)

    def luminosity_tev(self, t=None, energy_min=Quantity(1, "TeV")):
        """Gamma-ray luminosity above ``energy_min`` at age ``t``.

        The luminosity is assumed constant in a given age interval and zero
        before and after. The assumed spectral index is 2.1.

        Reference: http://adsabs.harvard.edu/abs/1994A%26A...287..959D (Formula (7)).

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.
        energy_min : `~astropy.units.Quantity`
            Lower energy limit for the luminosity.

        Notes
        -----
        The gamma-ray luminosity above 1 TeV is given by:

        .. math::

            L_{\\gamma}(\\geq 1TeV) \\approx 10^{34} \\theta
            \\left(\\frac{E_{SN}}{10^{51} erg}\\right)
            \\left(\\frac{\\rho_{ISM}}{1.66\\cdot 10^{-24} g/cm^{3}} \\right)
            \\textnormal{ph} s^{-1}

        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")

        # Flux in 1 k distance according to Drury formula 9
        term_0 = energy_min / Quantity(1, "TeV")
        term_1 = self.e_sn / Quantity(1e51, "erg")
        term_2 = self.rho_ISM / (Quantity(1, "cm-3") * const.m_p)
        L = self.theta * term_0 ** (1 - self.spectral_index) * term_1 * term_2

        # Corresponding luminosity
        L = np.select(
            [t <= self.sedov_taylor_begin, t <= self.sedov_taylor_end], [0, L]
        )
        return Quantity(1.0768E34, "s-1") * L

    @lazyproperty
    def sedov_taylor_begin(self):
        """Characteristic time scale when the Sedov-Taylor phase of the SNR's evolution begins.

        Notes
        -----
        The beginning of the Sedov-Taylor phase of the SNR is defined by the condition,
        that the swept up mass of the surrounding medium equals the mass of the
        ejected mass. The time scale is given by:

        .. math::

            t_{begin} \\approx 200 \\ \\textnormal{}
            \\left(\\frac{E_{SN}}{10^{51}erg}\\right)^{-1/2}
            \\left(\\frac{M_{ej}}{M_{\\odot}}\\right)^{5/6}
            \\left(\\frac{\\rho_{ISM}}{10^{-24}g/cm^3}\\right)^{-1/3}

        """
        term1 = (self.e_sn / Quantity(1e51, "erg")) ** (-1. / 2)
        term2 = (self.m_ejecta / const.M_sun) ** (5. / 6)
        term3 = (self.rho_ISM / (Quantity(1, "cm-3") * const.m_p)) ** (-1. / 3)
        return Quantity(200, "yr") * term1 * term2 * term3

    @lazyproperty
    def sedov_taylor_end(self):
        """Characteristic time scale when the Sedov-Taylor phase of the SNR's evolution ends.

        Notes
        -----
        The end of the Sedov-Taylor phase of the SNR is defined by the condition, that the
        temperature at the shock drops below T = 10^6 K. The time scale is given by:

        .. math::

            t_{end} \\approx 43000 \\textnormal{ }
            \\left(\\frac{m}{1.66\\cdot 10^{-24}g}\\right)^{5/6}
            \\left(\\frac{E_{SN}}{10^{51}erg}\\right)^{1/3}
            \\left(\\frac{\\rho_{ISM}}{1.66\\cdot 10^{-24}g/cm^3}\\right)^{-1/3}

        """
        term1 = 3 * const.m_p.cgs / (100 * const.k_B.cgs * self.t_stop)
        term2 = (self.e_sn / self.rho_ISM) ** (2. / 5)
        return ((term1 * term2) ** (5. / 6)).to("yr")


class SNRTrueloveMcKee(SNR):
    """SNR model according to Truelove & McKee (1999).

    Reference: http://adsabs.harvard.edu/abs/1999ApJS..120..299T
    """

    def __init__(self, *args, **kwargs):
        super(SNRTrueloveMcKee, self).__init__(*args, **kwargs)

        # Characteristic dimensions
        self.r_c = self.m_ejecta ** (1. / 3) * self.rho_ISM ** (-1. / 3)
        self.t_c = (
            self.e_sn ** (-1. / 2)
            * self.m_ejecta ** (5. / 6)
            * self.rho_ISM ** (-1. / 3)
        )

    def radius(self, t=None):
        """Outer shell radius at age t.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.

        Notes
        -----
        The radius during the free expansion phase is given by:

        .. math::

            R_{SNR}(t) = 1.12R_{ch}\\left(\\frac{t}{t_{ch}}\\right)^{2/3}

        The radius during the Sedov-Taylor phase evolves like:

        .. math::

            R_{SNR}(t) = \\left[R_{SNR, ST}^{5/2} + \\left(2.026\\frac{E_{SN}}
            {\\rho_{ISM}}\\right)^{1/2}(t - t_{ST})\\right]^{2/5}

        Using the characteristic dimensions:

        .. math::

            R_{ch} = M_{ej}^{1/3}\\rho_{ISM}^{-1/3} \\ \\
            \\textnormal{and} \\ \\ t_{ch} = E_{SN}^{-1/2}M_{ej}^{5/6}\\rho_{ISM}^{-1/3}

        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")

        # Evaluate `_radius_sedov_taylor` on `t > self.sedov_taylor_begin`
        # only to avoid a warning
        r = np.empty(t.shape, dtype=np.float64)
        mask = t > self.sedov_taylor_begin
        r[mask] = self._radius_sedov_taylor(t[mask]).to("cm").value
        r[~mask] = self._radius_free_expansion(t[~mask]).to("cm").value
        return Quantity(r, "cm")

    def _radius_free_expansion(self, t):
        """Shock radius at age t during free expansion phase.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.

        """
        return 1.12 * self.r_c * (t / self.t_c) ** (2. / 3)

    def _radius_sedov_taylor(self, t):
        """Shock radius  at age t during Sedov Taylor phase.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.
        """
        term1 = self._radius_free_expansion(self.sedov_taylor_begin) ** (5. / 2)
        term2 = (2.026 * (self.e_sn / self.rho_ISM)) ** (1. / 2)
        return (term1 + term2 * (t - self.sedov_taylor_begin)) ** (2. / 5)

    @lazyproperty
    def sedov_taylor_begin(self):
        """Characteristic time scale when the Sedov-Taylor phase starts.

        Given by :math:`t_{ST} \\approx 0.52 t_{ch}`.
        """
        return 0.52 * self.t_c

    def radius_reverse_shock(self, t):
        """Reverse shock radius at age t.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR.

        Notes
        -----
        Initially the reverse shock co-evolves with the radius of the SNR:

        .. math::

            R_{RS}(t) = \\frac{1}{1.19}r_{SNR}(t)

        After a time :math:`t_{core} \\simeq 0.25t_{ch}` the reverse shock reaches
        the core and then propagates as:

        .. math::

            R_{RS}(t) = \\left[1.49 - 0.16 \\frac{t - t_{core}}{t_{ch}} - 0.46
            \\ln \\left(\\frac{t}{t_{core}}\\right)\\right]\\frac{R_{ch}}{t_{ch}}t
        """
        if t is not None:
            validate_physical_type("t", t, "time")
        elif hasattr(self, "age"):
            t = self.age
        else:
            raise ValueError("Need time variable or age attribute.")

        # Time when reverse shock reaches the "core"
        t_core = 0.25 * self.t_c

        term1 = (t - t_core) / (self.t_c)
        term2 = 1.49 - 0.16 * term1 - 0.46 * np.log(t / t_core)
        R_1 = self._radius_free_expansion(t) / 1.19
        R_RS = term2 * (self.r_c / self.t_c) * t
        r = np.where(t < t_core, R_1.to("cm").value, R_RS.to("cm").value)
        return Quantity(r, "cm")

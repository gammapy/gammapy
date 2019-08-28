# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Supernova remnant (SNR) source models."""
import numpy as np
import astropy.constants
from astropy.units import Quantity
from astropy.utils import lazyproperty

__all__ = ["SNR", "SNRTrueloveMcKee"]


class SNR:
    """Simple supernova remnant (SNR) evolution model.

    The model is based on the Sedov-Taylor solution for strong explosions.

    Reference: https://ui.adsabs.harvard.edu/abs/1950RSPSA.201..159T

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
        Post-shock temperature where gamma-ray emission stops
    """

    def __init__(
        self,
        e_sn="1e51 erg",
        theta=Quantity(0.1),
        n_ISM=Quantity(1, "cm-3"),
        m_ejecta=astropy.constants.M_sun,
        t_stop=Quantity(1e6, "K"),
        age=None,
        morphology="Shell2D",
        spectral_index=2.1,
    ):
        self.e_sn = Quantity(e_sn, "erg")
        self.theta = theta
        self.rho_ISM = n_ISM * astropy.constants.m_p
        self.n_ISM = n_ISM
        self.m_ejecta = m_ejecta
        self.t_stop = t_stop
        self.morphology = morphology
        self.spectral_index = spectral_index
        if age is not None:
            self.age = Quantity(age, "yr")

    def radius(self, t):
        r"""Outer shell radius at age t.

        The radius during the free expansion phase is given by:

        .. math::
            r_{SNR}(t) \approx 0.01
            \left(\frac{E_{SN}}{10^{51}erg}\right)^{1/2}
            \left(\frac{M_{ej}}{M_{\odot}}\right)^{-1/2} t
            \text{ pc}

        The radius during the Sedov-Taylor phase evolves like:

        .. math::
            r_{SNR}(t) \approx \left(\frac{E_{SN}}{\rho_{ISM}}\right)^{1/5}t^{2/5}

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        t = Quantity(t, "yr")
        r = np.where(
            t > self.sedov_taylor_begin,
            self._radius_sedov_taylor(t).to_value("cm"),
            self._radius_free_expansion(t).to_value("cm"),
        )
        return Quantity(r, "cm")

    def _radius_free_expansion(self, t):
        """Shock radius at age t during free expansion phase.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        # proportional constant for the free expansion phase
        term_1 = (self.e_sn / Quantity(1e51, "erg")) ** (1.0 / 2)
        term_2 = (self.m_ejecta / astropy.constants.M_sun) ** (-1.0 / 2)
        return Quantity(0.01, "pc/yr") * term_1 * term_2 * t

    def _radius_sedov_taylor(self, t):
        """Shock radius  at age t  during Sedov Taylor phase.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        R_FE = self._radius_free_expansion(self.sedov_taylor_begin)
        return R_FE * (t / self.sedov_taylor_begin) ** (2.0 / 5)

    def radius_inner(self, t, fraction=0.0914):
        """Inner radius  at age t  of the SNR shell.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        return self.radius(t) * (1 - fraction)

    def luminosity_tev(self, t, energy_min="1 TeV"):
        r"""Gamma-ray luminosity above ``energy_min`` at age ``t``.

        The luminosity is assumed constant in a given age interval and zero
        before and after. The assumed spectral index is 2.1.

        The gamma-ray luminosity above 1 TeV is given by:

        .. math::
            L_{\gamma}(\geq 1TeV) \approx 10^{34} \theta
            \left(\frac{E_{SN}}{10^{51} erg}\right)
            \left(\frac{\rho_{ISM}}{1.66\cdot 10^{-24} g/cm^{3}} \right)
            \text{ s}^{-1}

        Reference: https://ui.adsabs.harvard.edu/abs/1994A%26A...287..959D (Formula (7)).

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        energy_min : `~astropy.units.Quantity`
            Lower energy limit for the luminosity
        """
        t = Quantity(t, "yr")
        energy_min = Quantity(energy_min, "TeV")

        # Flux in 1 k distance according to Drury formula 9
        term_0 = energy_min / Quantity(1, "TeV")
        term_1 = self.e_sn / Quantity(1e51, "erg")
        term_2 = self.rho_ISM / (Quantity(1, "cm-3") * astropy.constants.m_p)
        L = self.theta * term_0 ** (1 - self.spectral_index) * term_1 * term_2

        # Corresponding luminosity
        L = np.select(
            [t <= self.sedov_taylor_begin, t <= self.sedov_taylor_end], [0, L]
        )
        return Quantity(1.0768e34, "s-1") * L

    @lazyproperty
    def sedov_taylor_begin(self):
        r"""Characteristic time scale when the Sedov-Taylor phase of the SNR's evolution begins.

        The beginning of the Sedov-Taylor phase of the SNR is defined by the condition,
        that the swept up mass of the surrounding medium equals the mass of the
        ejected mass.

        The time scale is given by:

        .. math::
            t_{begin} \approx 200
            \left(\frac{E_{SN}}{10^{51}erg}\right)^{-1/2}
            \left(\frac{M_{ej}}{M_{\odot}}\right)^{5/6}
            \left(\frac{\rho_{ISM}}{10^{-24}g/cm^3}\right)^{-1/3}
            \text{yr}
        """
        term1 = (self.e_sn / Quantity(1e51, "erg")) ** (-1.0 / 2)
        term2 = (self.m_ejecta / astropy.constants.M_sun) ** (5.0 / 6)
        term3 = (self.rho_ISM / (Quantity(1, "cm-3") * astropy.constants.m_p)) ** (
            -1.0 / 3
        )
        return Quantity(200, "yr") * term1 * term2 * term3

    @lazyproperty
    def sedov_taylor_end(self):
        r"""Characteristic time scale when the Sedov-Taylor phase of the SNR's evolution ends.

        The end of the Sedov-Taylor phase of the SNR is defined by the condition, that the
        temperature at the shock drops below T = 10^6 K.

        The time scale is given by:

        .. math::
            t_{end} \approx 43000
            \left(\frac{m}{1.66\cdot 10^{-24}g}\right)^{5/6}
            \left(\frac{E_{SN}}{10^{51}erg}\right)^{1/3}
            \left(\frac{\rho_{ISM}}{1.66\cdot 10^{-24}g/cm^3}\right)^{-1/3}
            \text{yr}
        """
        term1 = (
            3
            * astropy.constants.m_p.cgs
            / (100 * astropy.constants.k_B.cgs * self.t_stop)
        )
        term2 = (self.e_sn / self.rho_ISM) ** (2.0 / 5)
        return ((term1 * term2) ** (5.0 / 6)).to("yr")


class SNRTrueloveMcKee(SNR):
    """SNR model according to Truelove & McKee (1999).

    Reference: https://ui.adsabs.harvard.edu/abs/1999ApJS..120..299T
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Characteristic dimensions
        self.r_c = self.m_ejecta ** (1.0 / 3) * self.rho_ISM ** (-1.0 / 3)
        self.t_c = (
            self.e_sn ** (-1.0 / 2)
            * self.m_ejecta ** (5.0 / 6)
            * self.rho_ISM ** (-1.0 / 3)
        )

    def radius(self, t):
        r"""Outer shell radius at age t.

        The radius during the free expansion phase is given by:

        .. math::
            R_{SNR}(t) = 1.12R_{ch}\left(\frac{t}{t_{ch}}\right)^{2/3}

        The radius during the Sedov-Taylor phase evolves like:

        .. math::
            R_{SNR}(t) = \left[R_{SNR, ST}^{5/2} + \left(2.026\frac{E_{SN}}
            {\rho_{ISM}}\right)^{1/2}(t - t_{ST})\right]^{2/5}

        Using the characteristic dimensions:

        .. math::
            R_{ch} = M_{ej}^{1/3}\rho_{ISM}^{-1/3} \ \
            \text{and} \ \ t_{ch} = E_{SN}^{-1/2}M_{ej}^{5/6}\rho_{ISM}^{-1/3}

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        t = Quantity(t, "yr")

        # Evaluate `_radius_sedov_taylor` on `t > self.sedov_taylor_begin`
        # only to avoid a warning
        r = np.empty(t.shape, dtype=np.float64)
        mask = t > self.sedov_taylor_begin
        r[mask] = self._radius_sedov_taylor(t[mask]).to_value("cm")
        r[~mask] = self._radius_free_expansion(t[~mask]).to_value("cm")
        return Quantity(r, "cm")

    def _radius_free_expansion(self, t):
        """Shock radius at age t during free expansion phase.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        return 1.12 * self.r_c * (t / self.t_c) ** (2.0 / 3)

    def _radius_sedov_taylor(self, t):
        """Shock radius  at age t during Sedov Taylor phase.

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        term1 = self._radius_free_expansion(self.sedov_taylor_begin) ** (5.0 / 2)
        term2 = (2.026 * (self.e_sn / self.rho_ISM)) ** (1.0 / 2)
        return (term1 + term2 * (t - self.sedov_taylor_begin)) ** (2.0 / 5)

    @lazyproperty
    def sedov_taylor_begin(self):
        r"""Characteristic time scale when the Sedov-Taylor phase starts.

        Given by :math:`t_{ST} \approx 0.52 t_{ch}`.
        """
        return 0.52 * self.t_c

    def radius_reverse_shock(self, t):
        r"""Reverse shock radius at age t.

        Initially the reverse shock co-evolves with the radius of the SNR:

        .. math::
            R_{RS}(t) = \frac{1}{1.19}r_{SNR}(t)

        After a time :math:`t_{core} \simeq 0.25t_{ch}` the reverse shock reaches
        the core and then propagates as:

        .. math::
            R_{RS}(t) = \left[1.49 - 0.16 \frac{t - t_{core}}{t_{ch}} - 0.46
            \ln \left(\frac{t}{t_{core}}\right)\right]\frac{R_{ch}}{t_{ch}}t

        Parameters
        ----------
        t : `~astropy.units.Quantity`
            Time after birth of the SNR
        """
        t = Quantity(t, "yr")

        # Time when reverse shock reaches the "core"
        t_core = 0.25 * self.t_c

        term1 = (t - t_core) / (self.t_c)
        term2 = 1.49 - 0.16 * term1 - 0.46 * np.log(t / t_core)
        R_1 = self._radius_free_expansion(t) / 1.19
        R_RS = term2 * (self.r_c / self.t_c) * t
        r = np.where(t < t_core, R_1.to_value("cm"), R_RS.to_value("cm"))
        return Quantity(r, "cm")

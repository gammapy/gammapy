# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""
import html
import numpy as np
import astropy.units as u

__all__ = ["JFactory"]


class JFactory:
    """Compute J-Factor or D-Factor maps.

    J-Factors are computed for annihilation and D-Factors for decay.
    Set the argument `annihilation` to `False` to compute D-Factors.
    The assumed dark matter profiles will be centered on the center of the map.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry.
    profile : `~gammapy.astro.darkmatter.profiles.DMProfile`
        Dark matter profile.
    distance : `~astropy.units.Quantity`
        Distance to convert angular scale of the map.
    annihilation: bool, optional
        Decay or annihilation. Default is True.
    """

    def __init__(self, geom, profile, distance, annihilation=True):
        self.geom = geom
        self.profile = profile
        self.distance = distance
        self.annihilation = annihilation

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def compute_differential_jfactor(self, ndecade=1e4):
        r"""Compute differential J-Factor.

        .. math::
            \frac{\mathrm d J_\text{ann}}{\mathrm d \Omega} =
            \int_{\mathrm{LoS}} \mathrm d l \rho(l)^2

        .. math::
            \frac{\mathrm d J_\text{decay}}{\mathrm d \Omega} =
            \int_{\mathrm{LoS}} \mathrm d l \rho(l)
        """
        separation = self.geom.separation(self.geom.center_skydir).rad
        rmin = u.Quantity(
            value=np.tan(separation) * self.distance, unit=self.distance.unit
        )
        rmax = self.distance
        val = [
            (
                2
                * self.profile.integral(
                    _.value * u.kpc,
                    rmax,
                    np.arctan(_.value / self.distance.value),
                    ndecade,
                    self.annihilation,
                )
                + self.profile.integral(
                    self.distance,
                    4 * rmax,
                    np.arctan(_.value / self.distance.value),
                    ndecade,
                    self.annihilation,
                )
            )
            for _ in rmin.ravel()
        ]
        integral_unit = u.Unit("GeV2 cm-5") if self.annihilation else u.Unit("GeV cm-2")
        jfact = u.Quantity(val).to(integral_unit).reshape(rmin.shape)
        return jfact / u.steradian

    def compute_jfactor(self, ndecade=1e4):
        r"""Compute astrophysical J-Factor.

        .. math::
            J(\Delta\Omega) =
           \int_{\Delta\Omega} \mathrm d \Omega^{\prime}
           \frac{\mathrm d J}{\mathrm d \Omega^{\prime}}
        """
        diff_jfact = self.compute_differential_jfactor(ndecade)
        return diff_jfact * self.geom.to_image().solid_angle()

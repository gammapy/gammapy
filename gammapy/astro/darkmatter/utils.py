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
    annihilation : bool, optional
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

        Parameters
        ----------
        ndecade : float, optional
            Number of sampling points per decade in radius used for the numerical
            integration. Default is 1e4.

        Returns
        -------
        jfactor : `~astropy.units.Quantity`
            Differential j-factor.

        Notes
        -----
        The line-of-sight (LoS) integral should include both the near and far
        sides of the halo. To account for this, the integration is split into
        two regions:

        1. :math:`[r_{\min}, r_{\max}]` - from the observer to the source,
           counted twice to include contributions from both near and far sides.
        2. :math:`[r_{\max}, 4 r_{\max}]` - from the source to infinity.
           The upper limit is truncated at :math:`4 r_{\max}` because
           contributions beyond this are negligible.

        Hence, the effective integration domain is:

        .. math::
            2 \times [r_{\min}, r_{\max}] \;+\; [r_{\max}, 4 r_{\max}].

        The LoS integral is converted into a radial integral over the profile through:

        .. math::
            r^2 = l^2 + r_{\max}^2 - 2 dl \cos \theta

        Rearranging for the differential gives:

        .. math::
            \mathrm dl = \frac{2 r}{\sqrt{r^2 - r_{\min}^2}} \, \mathrm dr.

        This substitution allows the integral to be evaluated directly as
        radial integrals using ``profile.integral``, giving

        .. math::
            \int_0^{l_\mathrm{max}} \rho^2(r(l, \theta)) \, \mathrm dl
            = 2 \int_{r_{\min}}^{r_{\max}} \frac{r \, \rho^2(r)}{\sqrt{r^2 - r_{\min}^2}} \, \mathrm dr
              + \int_{r_{\max}}^{4 r_{\max}} \frac{r \, \rho^2(r)}{\sqrt{r^2 - r_{\min}^2}} \, \mathrm dr.
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

        Parameters
        ----------
        ndecade : float, optional
            Number of sampling points per decade in radius used for the numerical
            integration. Default is 1e4.

        Returns
        -------
        jfactor : `~astropy.units.Quantity`
            The j-factor.
        """
        diff_jfact = self.compute_differential_jfactor(ndecade)
        return diff_jfact * self.geom.to_image().solid_angle()

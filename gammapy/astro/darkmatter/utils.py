# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""

import html

import astropy.units as u
from gammapy.modeling.models.prior import (
    LogNormalPrior,
)
import numpy as np

__all__ = ["JFactory", "add_factor_prior"]


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
        Distance from the observer to the dark matter halo center,
        used to compute the line-of-sight integration geometry.
    annihilation : `~astropy.units.Quantity`, optional
        Decay or annihilation. Default is True.
    rmax : `~astropy.units.Quantity`
        Physical size of the dark matter halo (upper limit of the
        line-of-sight integral). For extragalactic sources, this should
        be set to the halo radius (~kpc), **not** the distance to the
        source. Defaults to ``distance`` for backward compatibility,
        which is only appropriate for Galactic sources.
    """

    def __init__(self, geom, profile, distance, rmax, annihilation=True):
        self.geom = geom
        self.profile = profile
        self.distance = distance
        self.annihilation = annihilation
        self.rmax = rmax

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def _integrate_los_branch(self, impact, radius_min, radius_max, ndecade):
        """Integrate one radial line-of-sight branch."""
        exponent = 2 if self.annihilation else 1
        unit = radius_max.unit

        impact = impact.to(unit)
        radius_min = radius_min.to(unit)
        radius_max = radius_max.to(unit)

        if impact.value == 0:
            return self.profile.integral(
                radius_min, radius_max, 0, ndecade, self.annihilation, self.distance
            )

        logmin = np.log10(radius_min.value)
        logmax = np.log10(radius_max.value)
        n = max(2, int((logmax - logmin) * ndecade))

        t_min = np.arccosh(np.maximum((radius_min / impact).to_value(""), 1))
        t_max = np.arccosh(np.maximum((radius_max / impact).to_value(""), 1))
        t = np.linspace(t_min, t_max, n)

        radius = impact * np.cosh(t)
        values = self.profile(radius) ** exponent * radius

        return np.trapezoid(values, t)

    def _integrate_los(self, impact, separation, ndecade):
        """Integrate the physical forward line of sight."""
        distance = self.distance
        rmax = self.rmax

        if distance < rmax:
            integral = self._integrate_los_branch(impact, distance, rmax, ndecade)
            if separation < np.pi / 2:
                integral += 2 * self._integrate_los_branch(
                    impact, impact, distance, ndecade
                )
            return integral

        if separation < np.pi / 2 and impact < rmax:
            return 2 * self._integrate_los_branch(impact, impact, rmax, ndecade)

        return 0 * u.Unit("GeV2 cm-5" if self.annihilation else "GeV cm-2")

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
        The line-of-sight geometry is defined by

        .. math::
            r(l)^2 = D^2 + l^2 - 2 D l \cos\theta,

        where :math:`D` is the observer-to-halo-center distance and
        :math:`l \geq 0` is the physical forward line-of-sight coordinate.
        The impact parameter of the corresponding infinite line is given by:

        .. math::
            r_\perp = D \sin\theta.

        The integration is split into two regions:

        1. :math:`D < r_{\max}`: the observer is inside the integration radius.
        Directions with :math:`\theta < \pi / 2` cross the inner radial interval
        twice, while directions with :math:`\theta \geq \pi / 2` contain only
        the outward branch.

        2. :math:`D \geq r_{\max}`: the observer is outside the integration radius.
        The line of sight contributes only when it points toward the halo and
        intersects the integration sphere, i.e. when :math:`\theta < \pi / 2`
        and :math:`r_\perp < r_{\max}`.

        Each radial branch is evaluated using

        .. math::
            \mathrm dl =
            \frac{r}{\sqrt{r^2-r_\perp^2}}\,\mathrm dr.

        The apparent singularity at :math:`r = r_\perp` is integrable. To avoid
        evaluating it directly, each radial branch is integrated with the
        substitution :math:`r = r_\perp\cosh t`.
        """
        separation = self.geom.separation(self.geom.center_skydir).rad
        impact = u.Quantity(
            value=np.sin(separation) * self.distance, unit=self.distance.unit
        )
        val = [
            self._integrate_los(impact_i, separation_i, ndecade)
            for impact_i, separation_i in zip(impact.ravel(), separation.ravel())
        ]
        integral_unit = u.Unit("GeV2 cm-5") if self.annihilation else u.Unit("GeV cm-2")
        jfact = u.Quantity(val).to(integral_unit).reshape(impact.shape)
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


def add_factor_prior(model, sigma, mu=1.0):
    """Attach a Log Normal nuisance prior on ``scale`` for J/D-factor uncertainty.

    The J/D-factor is kept fixed at its nominal value; the associated
    uncertainty is instead expressed as an equivalent prior on ``scale``,
    since the predicted flux depends only on the product
    ``scale * jfactor``. Placing the prior directly on a second parameter
    (e.g. ``log10_jfactor``) would make it perfectly degenerate with
    ``scale``. This reparametrisation is a pure shift, so the prior
    retains the same shape and ``sigma``, centered at ``scale = 1``
    instead of at the nominal log10(J).

    Parameters
    ----------
    model : `~gammapy.astro.darkmatter.DarkMatterSpectralModel`
        Model whose ``scale`` parameter will get the prior attached.
        ``scale`` is unfrozen as part of this call.
    sigma : float
        Uncertainty on log10(J) (or log10(D)), in dex.
    mu : float, optional
        Center of the prior, in units of ``scale``. Default is 1.0, i.e.
        the nominal J/D-factor value.

    Returns
    -------
    model : `DarkMatterSpectralModel`
        The same model instance, with the prior attached, for chaining.
    """
    model.scale.frozen = False
    model.scale.prior = LogNormalPrior(mu=mu, sigma=sigma * np.log(10))
    return model

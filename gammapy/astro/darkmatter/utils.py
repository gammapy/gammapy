# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""

import html
import numbers
import warnings

import astropy.units as u
from gammapy.modeling.models.prior import LogSpaceGaussianPrior
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

    def _integrate_los(self, impact, separation, ndecade):
        """Integrate the physical forward line of sight."""
        distance = self.distance
        rmax = self.rmax
        exponent = 2 if self.annihilation else 1
        integral_unit = u.Unit("GeV2 cm-5" if self.annihilation else "GeV cm-2")

        if distance >= rmax and np.cos(separation) <= 0:
            return 0 * integral_unit

        discriminant = rmax**2 - impact**2
        if discriminant <= 0 * rmax.unit**2:
            return 0 * integral_unit

        los_half_length = np.sqrt(discriminant)
        observer = -distance * np.cos(separation)
        s_min = np.maximum(-los_half_length, observer)
        s_max = los_half_length

        if impact.value == 0:
            if s_min <= 0 * s_min.unit <= s_max:
                raise ValueError(
                    "The zero-impact central line of sight must be handled by "
                    "the central-pixel integration."
                )

            unit = s_min.unit
            radial_decades = np.log10((s_max / s_min).to_value(""))
            n = max(2, int(radial_decades * ndecade))
            s = (
                np.geomspace(
                    s_min.to_value(unit),
                    s_max.to_value(unit),
                    n,
                )
                * unit
            )

            return np.trapezoid(self.profile(s) ** exponent, s)

        t_min = np.arcsinh((s_min / impact).to_value(""))
        t_max = np.arcsinh((s_max / impact).to_value(""))

        radius_start = np.sqrt(impact**2 + s_min**2)
        radius_end = np.sqrt(impact**2 + s_max**2)
        if s_min <= 0 * s_min.unit <= s_max:
            radius_closest = impact
        else:
            radius_closest = np.minimum(radius_start, radius_end)

        radial_decades = np.log10(
            (radius_start / radius_closest).to_value("")
        ) + np.log10((radius_end / radius_closest).to_value(""))
        n = max(2, int(radial_decades * ndecade))

        t = np.linspace(t_min, t_max, max(n, 3))
        if t_min < 0 < t_max:
            t = np.union1d(t, [0.0])

        radius = impact * np.cosh(t)
        values = self.profile(radius) ** exponent * radius

        return np.trapezoid(values, t)

    def _integrate_central_pixel(self, ndecade, rtol, max_subdivision):
        """Adaptively average the line-of-sight integral over the central pixel."""
        central_geom = self.geom.to_image().cutout(
            position=self.geom.center_skydir,
            width=self.geom.pixel_scales,
        )
        parent_solid_angle = central_geom.solid_angle().sum()
        previous = None

        factor = 2
        while factor <= max_subdivision:
            # Even factors ensure no subpixel center coincides with the halo center.
            subgeom = central_geom.upsample(factor)
            separation = subgeom.separation(self.geom.center_skydir).rad
            impact = u.Quantity(
                value=np.sin(separation) * self.distance,
                unit=self.distance.unit,
            )

            values = [
                self._integrate_los(impact_i, separation_i, ndecade)
                for impact_i, separation_i in zip(impact.ravel(), separation.ravel())
            ]
            values = u.Quantity(values).reshape(impact.shape)
            current = (values * subgeom.solid_angle()).sum() / parent_solid_angle

            if previous is not None:
                relative_change = np.abs((current - previous) / current)
                if relative_change < rtol:
                    return current

            previous = current
            factor *= 2

        warnings.warn(
            "The requested central-pixel relative tolerance "
            f"(central_pixel_rtol={rtol}) was not reached "
            "before the maximum subdivision factor "
            f"(central_pixel_max_subdivision={max_subdivision}). "
            "Returning the finest estimate. Increase "
            "central_pixel_max_subdivision if higher accuracy is required.",
            UserWarning,
            stacklevel=3,
        )
        return previous

    def compute_differential_jfactor(
        self,
        ndecade=1e4,
        *,
        central_pixel_rtol=0.01,
        central_pixel_max_subdivision=128,
    ):
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
        central_pixel_rtol : float, optional
            Relative tolerance for adaptive refinement of the central pixel.
            Default is 0.01.
        central_pixel_max_subdivision : int, optional
            Maximum subdivision factor for adaptive refinement of the central
            pixel. Must be a power of two greater than or equal to 4. Default is
            128.

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

        Defining the signed coordinate

        .. math::
            s = l - D \cos\theta,

        gives :math:`r^2 = r_\perp^2 + s^2`. The physical forward ray
        (:math:`l \geq 0`) and the integration sphere
        (:math:`r \leq r_{\max}`) define the limits

        .. math::
            s_{\min} = \max\left(-\sqrt{r_{\max}^2-r_\perp^2},
            -D\cos\theta\right), \qquad
            s_{\max} = \sqrt{r_{\max}^2-r_\perp^2}.

        If these domains do not overlap, the integral is zero. Otherwise, for
        non-zero impact parameter, the line of sight is evaluated continuously
        using

        .. math::
            s = r_\perp\sinh t, \qquad
            r = r_\perp\cosh t, \qquad
            \mathrm ds = r\,\mathrm dt.

        A zero-impact line of sight that does not cross the halo center is
        integrated directly in positive :math:`s`. The central direction is
        handled by the adaptive central-pixel integration.
        """
        if not np.isfinite(central_pixel_rtol) or central_pixel_rtol <= 0:
            raise ValueError("central_pixel_rtol must be a finite positive number.")

        if (
            not isinstance(central_pixel_max_subdivision, numbers.Integral)
            or central_pixel_max_subdivision < 4
            or central_pixel_max_subdivision & (central_pixel_max_subdivision - 1)
        ):
            raise ValueError(
                "central_pixel_max_subdivision must be a power of two greater "
                "than or equal to 4."
            )

        separation = self.geom.separation(self.geom.center_skydir).rad
        impact = u.Quantity(
            value=np.sin(separation) * self.distance, unit=self.distance.unit
        )
        zero_separation = separation == 0
        if np.any(zero_separation):
            warnings.warn(
                "A pixel center coincides with the halo center, as can occur for "
                "a centered map with an odd number of pixels. The central pixel "
                "will be adaptively subdivided to avoid the zero-impact line of "
                "sight. Consider using an even number of pixels if this special "
                "treatment is not desired.",
                UserWarning,
                stacklevel=2,
            )

        val = [
            self._integrate_central_pixel(
                ndecade,
                rtol=central_pixel_rtol,
                max_subdivision=central_pixel_max_subdivision,
            )
            if is_zero_separation
            else self._integrate_los(impact_i, separation_i, ndecade)
            for impact_i, separation_i, is_zero_separation in zip(
                impact.ravel(), separation.ravel(), zero_separation.ravel()
            )
        ]
        integral_unit = u.Unit("GeV2 cm-5") if self.annihilation else u.Unit("GeV cm-2")
        jfact = u.Quantity(val).to(integral_unit).reshape(impact.shape)
        return jfact / u.steradian

    def compute_jfactor(
        self,
        ndecade=1e4,
        *,
        central_pixel_rtol=0.01,
        central_pixel_max_subdivision=128,
    ):
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
        central_pixel_rtol : float, optional
            Relative tolerance for adaptive refinement of the central pixel.
            Default is 0.01.
        central_pixel_max_subdivision : int, optional
            Maximum subdivision factor for adaptive refinement of the central
            pixel. Must be a power of two greater than or equal to 4. Default is
            128.

        Returns
        -------
        jfactor : `~astropy.units.Quantity`
            The j-factor.
        """
        diff_jfact = self.compute_differential_jfactor(
            ndecade,
            central_pixel_rtol=central_pixel_rtol,
            central_pixel_max_subdivision=central_pixel_max_subdivision,
        )
        return diff_jfact * self.geom.to_image().solid_angle()


def add_factor_prior(model, sigma):
    """Attach a log-normal nuisance prior to the astrophysical factor.

    The J/D-factor carries a multiplicative uncertainty of ``sigma`` dex from
    the stellar-kinematics analysis. This function attaches the corresponding
    log-normal prior to the ``factor`` parameter, which is unfrozen as part of
    the call, and leaves ``scale`` free and unconstrained.

    ``scale`` and ``factor`` are exactly degenerate: the predicted flux depends
    only on their product. That degeneracy is inherent to the measurement, and
    the prior is what curves the otherwise flat direction. Profiling ``factor`` at
    fixed ``scale`` is the standard treatment of this nuisance in a profile-likelihood analysis
    (Ackermann et al. 2014).

    Parameters
    ----------
    model : `~gammapy.astro.darkmatter.DarkMatterSpectralModel`
        Model whose ``factor`` parameter will get the prior attached.
    sigma : float
        Uncertainty on log10 of the astrophysical factor, in dex.

    Returns
    -------
    model : `DarkMatterSpectralModel`
        The same model instance, with the prior attached, for chaining.

    References
    ----------
    * Ackermann et al. (2014), Phys. Rev. D 89, 042001
    """
    nominal = model.factor.value if model.factor.value is not None else None
    model.factor.min = np.finfo(float).tiny
    model.factor.frozen = False
    model.factor.prior = LogSpaceGaussianPrior(mu=nominal, sigma=sigma * np.log(10))
    return model

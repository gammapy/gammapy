# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""
import astropy.units as u

__all__ = ["JFactory"]


class JFactory:
    """Compute J-Factor maps.

    All J-Factors are computed for annihilation. The assumed dark matter
    profiles will be centered on the center of the map.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry
    profile : `~gammapy.astro.darkmatter.profiles.DMProfile`
        Dark matter profile
    distance : `~astropy.units.Quantity`
        Distance to convert angular scale of the map
    """

    def __init__(self, geom, profile, distance):
        self.geom = geom
        self.profile = profile
        self.distance = distance

    def compute_differential_jfactor(self):
        r"""Compute differential J-Factor.

        .. math::
            \frac{\mathrm d J}{\mathrm d \Omega} =
            \int_{\mathrm{LoS}} \mathrm d r \rho(r)
        """
        # TODO: Needs to be implemented more efficiently
        separation = self.geom.separation(self.geom.center_skydir)
        rmin = separation.rad * self.distance
        rmax = self.distance
        val = [self.profile.integral(_, rmax) for _ in rmin.flatten()]
        jfact = u.Quantity(val).to("GeV2 cm-5").reshape(rmin.shape)
        return jfact / u.steradian

    def compute_jfactor(self):
        r"""Compute astrophysical J-Factor.

        .. math::
            J(\Delta\Omega) =
           \int_{\Delta\Omega} \mathrm d \Omega^{\prime}
           \frac{\mathrm d J}{\mathrm d \Omega^{\prime}}
        """
        diff_jfact = self.compute_differential_jfactor()
        return diff_jfact * self.geom.to_image().solid_angle()

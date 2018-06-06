# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to compute J-factor maps
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ...cube import make_separation_map
from ...maps import WcsNDMap
import astropy.units as u
import numpy as np
import copy


__all__ = [
    'JFactory',
    'DMFluxMapMaker',
]


class JFactory(object):
    """Compute J-Factor maps

    All J-Factors are computed for annihilation. The assumend dark matter
    profiles will be centered on the center of the map

    Parameters
    ----------
    map_ : `~gammapy.maps.WcsMap`
        Sky map
    profile : `~gammapy.astro.darkmatter.profiles.DMProfile`
        Dark matter profile
    distance : `~astropy.units.Quantity`
        Distance to convert angular scale of the map
    """

    def __init__(self, map_, distance, profile):
        self.profile = profile
        self.map_ = map_
        self.distance = distance

    @property
    def center(self):
        """Center of the profile"""
        return self.map_.geom.center_skydir

    def run(self):
        """Run all steps"""
        self.compute_differential_jfactor()
        self.compute_jfactor()

    def compute_differential_jfactor(self):
        r"""Compute differential J-Factor

        .. math ::
            \frac{\mathrm d J}{\mathrm d \Omega} = 
           \int_{\mathrm{LoS}} \mathrm d r \rho(r)


        TODO: Needs to be implemented more efficiently
        """
        angular_dist = make_separation_map(
            geom=self.map_.geom,
            position=self.center)
        rmin = angular_dist.quantity.to('rad').data * self.distance
        rmax = self.distance
        val = [self.profile.integral(_, rmax) for _ in rmin.flatten()]
        jfact = u.Quantity(val).to('GeV2 cm-5').reshape(rmin.shape)
        self.diff_jfact = WcsNDMap(data=jfact.value,
                                   geom=self.map_.geom,
                                   unit=jfact.unit / u.steradian)

    def compute_jfactor(self):
        r"""Compute astrophysical J-Factor

        .. math ::
            J(\Delta\Omega) = 
           \int_{\Delta\Omega} \mathrm d \Omega^{\prime}
           \frac{\mathrm d J}{\mathrm d \Omega^{\prime}} 

        """
        jfact = self.diff_jfact.quantity
        jfact *= self.map_.geom.to_image().solid_angle()

        self.jfact = WcsNDMap(data=jfact,
                              geom=self.map_.geom)


class DMFluxMapMaker(object):
    r"""Create dark matter flux maps

    The gamma-ray flux is computed as follows

    .. math::

        \frac{\mathrm d \phi}{\mathrm d E \mathrm d\Omega} =
        \frac{\langle \sigma\nu \rangle}{8\pi m^2_{\mathrm{DM}}}
        \frac{\mathrm d N}{\mathrm dE} \times J(\Delta\Omega)

    Parameters
    ----------
    jfact_map : `~gammapy.maps.WcsNDMap`
        J-Factor map as computed by `~gammapy.astro.darkmatter.JFactory`
    prim_flux : `~gammapy.astro.darkmatter.PrimaryFlux`
        Primary gamma-ray flux
    x_section : `~astropy.units.Quantity`
        Velocity averaged annihilation cross section, $\langle \sigma\nu\rangle$
    energy_range : tuple of `~astropy.units.Quantity`
        Energy range for the map

    References
    ----------
    * `2011JCAP...03..051 <http://adsabs.harvard.edu/abs/2011JCAP...03..051>`_
    """

    def __init__(self, jfact_map, prim_flux, x_section, energy_range):
        self.jfact_map = jfact_map
        self.prim_flux = prim_flux
        self.x_section = x_section
        self.energy_range = energy_range
        self._flux_map = None

    def run(self):
        """Make the map

        The result is stored in the
        :func:`~gammapy.astro.darkmatter.DMFluxMapMaker.flux_map` attribute
        """
        prefactor = (self.x_section / (8 * np.pi * self.prim_flux.mDM ** 2))
        int_flux = self.prim_flux.table_model.integral(
            emin=self.energy_range[0],
            emax=self.energy_range[1],
        )
        data = self.jfact_map.quantity.copy()
        data *= prefactor
        data *= int_flux
        data = data.to('cm-2 s-1')

        self._fluxmap = WcsNDMap(data=data, geom=self.jfact_map.geom)

    @property
    def flux_map(self):
        """Flux map"""
        return self._fluxmap

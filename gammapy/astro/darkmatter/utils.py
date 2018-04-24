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
]

class JFactory(object):
    """Helper class to compute J-Factor maps

    All J-Factors are computed for annihilation. The assumend dark matter
    profiles will be centered on the center of the map
    
    Parameters
    ----------
    map_ : `~gammapy.maps.WcsMap`
        Sky map
    profile : `~gammapy.astro.darkmatter.DMProfile`
        Dark matter profile
    distance : `~astropy.units.Quantity`
        Distance to convert angular scale of the map
    """
    def __init__(self, map_, distance, profile):
        self.profile=profile
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

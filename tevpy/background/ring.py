# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Ring background estimation

Implements a simple RingBgMaker for cartesian coordinates.
TODO: Add psi cut
"""
from __future__ import division
import numpy as np
from ..image.utils import ring_correlate

__all__ = ['RingBgMaker', 'outer_ring_radius', 'area_factor', 'alpha']


class RingBgMaker:
    """Ring background method for cartesian coordinates.
    Step 1: apply exclusion mask
    Step 2: ring-correlate
    Step 3: apply psi cut
    @todo psi cut"""
    def __init__(self, r_i, r_o, pixscale=0.01):
        """Initialize the RingBgMaker.
        @param r_i: inner ring radius (deg)
        @param r_o: outer ring radius (deg)
        @param pixscale: degrees per pixel"""
        self.pixscale = float(pixscale)
        # Note: internally all computations are in pixels,
        # so convert deg to pix here:
        self.r_i = r_i / self.pixscale
        self.r_o = r_o / self.pixscale

    def info(self):
        """Print some basic parameter info."""
        print('RingBgMaker parameters:')
        print('r_i: %g pix = %g deg'.format(
            (self.r_i, self.r_i * self.pixscale)))
        print('r_o: %g pix = %g deg'.format(
            (self.r_o, self.r_o * self.pixscale)))
        print('pixscale: %g deg/pix'.format(self.pixscale))
        print()

    def correlate(self, image):
        """Correlate a given image with the ring."""
        return ring_correlate(image, self.r_i, self.r_o)

    def correlate_maps(self, maps):
        """Compute off maps from on maps by correlating with the ring,
        taking the exclusion map into account.
        maps: maps.Maps object"""
        # Note: maps['on'] returns a copy of the HDU,
        # so assigning to on would be pointless.
        n_on = maps['n_on'].data
        a_on = maps['a_on'].data
        exclusion = maps['exclusion'].data
        maps['n_off'].data = self.correlate(n_on * exclusion)
        maps['a_off'].data = self.correlate(a_on * exclusion)
        maps.is_off_correlated = True


def outer_ring_radius(theta, inner_ring_radius, area_factor):
    """Compute outer ring radius
    @param theta: on region radius
    @param inner_ring_radius: inner ring radius
    @param area_factor: desired off / on area ratio

    The determining equation is:
    area_factor = off_area / on_area =
    (pi (r_o**2 - r_i**2)) / (pi * theta**2 )
    """
    return np.sqrt(area_factor * theta ** 2 + inner_ring_radius ** 2)


def area_factor(theta, inner_ring_radius, outer_ring_radius):
    """Compute areafactor.
    @param theta: on region radius
    @param r_i: inner ring radius
    @param r_0: outer ring radius"""
    return (outer_ring_radius ** 2 - inner_ring_radius ** 2) / theta ** 2


def alpha(theta, inner_ring_radius, outer_ring_radius):
    """Compute alpha, the inverse area factor.
    @param theta: on region radius
    @param r_i: inner ring radius
    @param r_0: outer ring radius"""
    return 1. / area_factor(theta, inner_ring_radius, outer_ring_radius)

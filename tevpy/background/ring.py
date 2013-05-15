"""
Ring background estimation

Implements a simple RingBgMaker for cartesian coordinates.
TODO: Add psi cut
"""
import numpy as np
from image.utils import ring_correlate

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
        print 'RingBgMaker parameters:'
        print 'r_i: %g pix = %g deg' % \
            (self.r_i, self.r_i * self.pixscale)
        print 'r_o: %g pix = %g deg' % \
            (self.r_o, self.r_o * self.pixscale)
        print 'pixscale: %g deg/pix' % self.pixscale
        print

    def correlate(self, image):
        """Correlate a given image with the ring."""
        return ring_correlate(image, self.r_i, self.r_o)

    def correlate_maps(self, maps):
        """Compute off maps from on maps by correlating with the ring,
        taking the exclusion map into account.
        maps: bgmaps.BgMaps object"""
        # Note: maps['on'] returns a copy of the HDU,
        # so assigning to on would be pointless.
        on = maps['on'].data
        onexposure = maps['onexposure'].data
        exclusion = maps['exclusion'].data
        maps['off'].data = self.correlate(on * exclusion)
        maps['offexposure'].data = self.correlate(onexposure * exclusion)
        maps.is_off_correlated = True


def outer_ring_radius(theta, inner_ring_radius, areafactor):
    """Compute outer ring radius
    @param theta: on region radius
    @param inner_ring_radius: inner ring radius
    @param areafactor: desired off / on area ratio

    The determining equation is:
    areafactor = offarea / onarea =
    (pi (r_o**2 - r_i**2)) / (pi * theta**2 )
    """
    return np.sqrt(areafactor * theta ** 2 + inner_ring_radius ** 2)


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

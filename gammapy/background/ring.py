# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Ring background estimation.
"""
from __future__ import print_function, division
import numpy as np
from ..image import ring_correlate

__all__ = ['ring_correlate_off_maps', 'RingBgMaker',
           'ring_r_out', 'ring_area_factor', 'ring_alpha']


class RingBgMaker(object):
    """Ring background method for cartesian coordinates.

    Step 1: apply exclusion mask
    Step 2: ring-correlate
    Step 3: apply psi cut

    TODO: add method to apply the psi cut

    Parameters
    ----------
    r_in : float
        Inner ring radius (deg)
    r_out : float
        Outer ring radius (deg)
    pixscale : float
        degrees per pixel
    """
    def __init__(self, r_in, r_out, pixscale=0.01):
        self.pixscale = float(pixscale)
        # Note: internally all computations are in pixels,
        # so convert deg to pix here:
        self.r_in = r_in / self.pixscale
        self.r_out = r_out / self.pixscale

    def info(self):
        """Print some basic parameter info."""
        print('RingBgMaker parameters:')
        fmt = 'r_in: {0} pix = {1} deg'
        print(fmt.format(self.r_in, self.r_in * self.pixscale))
        fmt = 'r_out: {0} pix = {1} deg'
        print(fmt.format(self.r_out, self.r_out * self.pixscale))
        print('pixscale: {0} deg/pix'.format(self.pixscale))
        print()

    def correlate(self, image):
        """Ring-correlate a given image."""
        return ring_correlate(image, self.r_in, self.r_out)

    def correlate_maps(self, maps):
        """Compute off maps as ring-correlated versions of the on maps.

        The exclusion map is taken into account.

        Parameters
        ----------
        maps : gammapy.background.maps.Maps
            Input maps (is modified in-place)
        """
        # Note: maps['on'] returns a copy of the HDU,
        # so assigning to on would be pointless.
        n_on = maps['n_on'].data
        a_on = maps['a_on'].data
        exclusion = maps['exclusion'].data
        maps['n_off'].data = self.correlate(n_on * exclusion)
        maps['a_off'].data = self.correlate(a_on * exclusion)
        maps.is_off_correlated = True


def ring_correlate_off_maps(maps, r_in, r_out):
    """Ring-correlate the basic off maps.

    Parameters
    ----------
    maps : gammapy.background.maps.Maps
        Maps container
    r_in : float
        Inner ring radius (deg)
    r_out : float
        Outer ring radius (deg)
    """
    pixscale = maps['n_on'].header['CDELT2']
    ring_bg_maker = RingBgMaker(r_in, r_out, pixscale)
    return ring_bg_maker.correlate_maps(maps)


def ring_r_out(theta, r_in, area_factor):
    """Compute ring outer radius.

    The determining equation is:
        area_factor =
        off_area / on_area =
        (pi (r_out**2 - r_in**2)) / (pi * theta**2 )

    Parameters
    ----------
    theta : float
        On region radius
    r_in : float
        Inner ring radius
    area_factor : float
        Desired off / on area ratio

    Returns
    -------
    r_out : float
        Outer ring radius
    """
    return np.sqrt(area_factor * theta ** 2 + r_in ** 2)


def ring_area_factor(theta, r_in, r_out):
    """Compute ring area factor.

    Parameters
    ----------
    theta : float
        On region radius
    r_in : float
        Inner ring radius
    r_out : float
        Outer ring radius
    """
    return (r_out ** 2 - r_in ** 2) / theta ** 2


def ring_alpha(theta, r_in, r_out):
    """Compute ring alpha, the inverse area factor.

    Parameters
    ----------
    theta : float
        On region radius
    r_in : float
        Inner ring radius
    r_out : float
        Outer ring radius
    """
    return 1. / ring_area_factor(theta, r_in, r_out)

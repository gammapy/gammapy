# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from photutils.utils.wcs_helpers import (
    skycoord_to_pixel_scale_angle,
    assert_angle_or_pixel,
    skycoord_to_pixel_mode
)
from .core import SkyRegion, PixRegion

__all__ = [
    'PixCircleRegion',
    'SkyCircleRegion',
]


class PixCircleRegion(PixRegion):
    pass


class SkyCircleRegion(SkyRegion):
    """
    Circular aperture(s), defined in sky coordinates.

    Parameters
    ----------
    pos : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be either
        scalar coordinates or an array of coordinates.
    radius : `~astropy.units.Quantity`
        The radius of the aperture(s), either in angular or pixel units.
    """

    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

    def to_pixel(self, wcs):
        """
        Return a CircularAperture instance in pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            WCS object
        """

        x, y = skycoord_to_pixel(self.pos, wcs, mode=skycoord_to_pixel_mode)

        central_pos = SkyCoord([wcs.wcs.crval], frame=self.pos.name, unit=wcs.wcs.cunit)
        xc, yc, scale, angle = skycoord_to_pixel_scale_angle(central_pos, wcs)
        pix_radius = (scale * self.radius).to(u.pixel).value

        pix_position = np.array([x, y]).transpose()

        return PixCircleRegion(pix_position, pix_radius)

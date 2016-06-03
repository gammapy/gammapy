import math

import numpy as np
from astropy import units as u

from ..core import PixelRegion, SkyRegion


class PointPixelRegion(PixelRegion):
    """
    A point position in pixel coordinates.

    Parameters
    ----------
    center : :class:`~regions.core.pixcoord.PixCoord`
        The position of the point
    """

    def __init__(self, center, meta=None, visual=None):
        # TODO: test that center is a 0D PixCoord
        self.center = center
        self.meta = meta or {}
        self.visual = visual or {}

    @property
    def area(self):
        return 0

    def contains(self, pixcoord):
        return False

    def to_shapely(self):
        return self.center.to_shapely()

    def to_sky(self, wcs, mode='local', tolerance=None):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def to_mask(self, mode='center'):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def as_patch(self, **kwargs):
        # TODO: needs to be implemented
        raise NotImplementedError("")



class PointSkyRegion(SkyRegion):
    """
    A pixel region in sky coordinates.

    Parameters
    ----------
    center : :class:`~astropy.coordinates.SkyCoord`
        The position of the point
    """

    def __init__(self, center, meta=None, visual=None):
        # TODO: test that center is a 0D SkyCoord
        self.center = center
        self.meta = meta or {}
        self.visual = visual or {}

    @property
    def area(self):
        return 0

    def contains(self, skycoord):
        return False

    def to_pixel(self, wcs, mode='local', tolerance=None):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def as_patch(self, **kwargs):
        # TODO: needs to be implemented
        raise NotImplementedError("")


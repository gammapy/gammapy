# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ..core import PixelRegion, SkyRegion

__all__ = ['PolygonPixelRegion', 'PolygonSkyRegion']


class PolygonPixelRegion(PixelRegion):
    """
    A polygon in pixel coordinates.

    Parameters
    ----------
    vertices : :class:`~regions.core.pixcoord.PixCoord`
        The vertices of the polygon
    """

    def __init__(self, vertices, meta=None, visual=None):
        # TODO: test that vertices is a 1D PixCoord
        self.vertices = vertices
        self.meta = meta or {}
        self.visual = visual or {}

    @property
    def area(self):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def contains(self, pixcoord):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def to_shapely(self):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def to_sky(self, wcs, mode='local', tolerance=None):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def to_mask(self, mode='center'):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def as_patch(self, **kwargs):
        # TODO: needs to be implemented
        raise NotImplementedError("")


class PolygonSkyRegion(SkyRegion):
    """
    A polygon in sky coordinates.

    Parameters
    ----------
    vertices : :class:`~regions.core.pixcoord.PixCoord`
        The vertices of the polygon
    """

    def __init__(self, vertices, meta=None, visual=None):
        # TODO: test that vertices is a 1D SkyCoord
        self.vertices = vertices
        self.meta = meta or {}
        self.visual = visual or {}

    @property
    def area(self):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def contains(self, skycoord):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def to_pixel(self, wcs, mode='local', tolerance=None):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def as_patch(self, **kwargs):
        # TODO: needs to be implemented
        raise NotImplementedError("")

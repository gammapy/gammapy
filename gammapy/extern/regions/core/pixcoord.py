# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ['PixCoord']


class PixCoord(object):
    """
    Class representing a collection of pixel coordinates
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x
        yield self.y

    def to_shapely(self):
        from shapely.geometry import Point
        return Point(self.x, self.y)

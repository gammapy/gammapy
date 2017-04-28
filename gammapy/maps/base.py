# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [
    'MapBase',
]


class MapBase(object):
    """Abstract map class.

    This can represent either WCS or HEALPIX-based maps in 2 or 3 dimensions.
    """

    def __init__(self, geom, data):
        self._data = data
        self._geom = geom

    @property
    def data(self):
        return self._data

    @property
    def geom(self):
        return self._geom

    @data.setter
    def data(self, val):
        if val.shape != self.data.shape:
            raise Exception('Wrong shape.')
        self._counts = val

    def sum_over_axes(self):
        """Reduce a counts cube to a counts map by summing over the energy planes."""
        pass

    def get_by_coord(self, coord, interp=None):
        """Return the map values corresponding to a set of map coordinates."""
        pass

    def get_by_pix(self, pix, interp=None):
        """Return the map values corresponding to a set of pixel coordinates."""
        pass

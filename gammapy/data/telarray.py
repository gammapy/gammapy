# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.table import Table
from . import utils

__all__ = ['TelescopeArray']


class TelescopeArray(Table):
    """Telescope array info.
    """
    @property
    def info(self):
        """Summary info string."""
        s = '---> Telescope array info:\n'
        s += '- number of telescopes: {}\n'.format(len(self))
        return s

    @property
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)"""
        return utils._earth_location_from_dict(self.meta)

    def plot(self, ax):
        """Plot telescope locations."""
        raise NotImplementedError

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table
from .utils import _earth_location_from_dict

__all__ = [
    'TelescopeArray',
]


class TelescopeArray(Table):
    """Telescope array info.

    TODO: is this available in ctapipe?
    """
    @property
    def summary(self):
        """Summary info string."""
        s = '---> Telescope array info:\n'
        s += '- number of telescopes: {}\n'.format(len(self))
        return s

    @property
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)"""
        return _earth_location_from_dict(self.meta)

    def plot(self, ax):
        """Plot telescope locations."""
        raise NotImplementedError

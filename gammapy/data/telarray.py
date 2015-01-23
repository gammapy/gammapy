# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.table import Table
from astropy.coordinates import Angle, EarthLocation
from astropy.units import Quantity

__all__ = ['TelescopeArray']


class TelescopeArray(Table):
    """Telescope array info.
    """
    def __str__(self):
        # TODO: implement useful info
        return str(self)

    def get_earth_location(self):
        """Array center `~astropy.coordinates.EarthLocation`.
        """
        meta = self.meta
        lon = Angle(meta['GEOLON'], 'deg')
        lat = Angle(meta['GEOLAT'], 'deg')
        height = Quantity(meta['ALTITUDE'], 'meter')
        return EarthLocation(lon=lon, lat=lat, height=height)

    def plot(self, ax):
        """Plot telescope locations."""
        raise NotImplementedError

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Misc utility functions."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.coordinates import Angle, EarthLocation
from astropy.units import Quantity


def _earth_location_from_dict(meta):
    lon = Angle(meta['GEOLON'], 'deg')
    lat = Angle(meta['GEOLAT'], 'deg')
    height = Quantity(meta['ALTITUDE'], 'meter')
    return EarthLocation(lon=lon, lat=lat, height=height)

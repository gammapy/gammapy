# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Misc utility functions."""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.coordinates import Angle, EarthLocation
from astropy.units import Quantity

__all__ = [

]


def _earth_location_from_dict(meta):
    """Create `~astropy.coordinates.EarthLocation` from FITS header dict."""
    lon = Angle(meta['GEOLON'], 'deg')
    lat = Angle(meta['GEOLAT'], 'deg')
    # TODO: should we support both here?
    # Check latest spec if ALTITUDE is used somewhere.
    if 'GEOALT' in meta:
        height = Quantity(meta['GEOALT'], 'meter')
    elif 'ALTITUDE' in meta:
        height = Quantity(meta['ALTITUDE'], 'meter')
    else:
        raise KeyError('The GEOALT or ALTITUDE header keyword must be set')

    return EarthLocation(lon=lon, lat=lat, height=height)

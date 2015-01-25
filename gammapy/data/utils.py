# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Misc utility functions."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.time import Time
from astropy.coordinates import Angle, EarthLocation
from astropy.units import Quantity


def _earth_location_from_dict(meta):
    lon = Angle(meta['GEOLON'], 'deg')
    lat = Angle(meta['GEOLAT'], 'deg')
    height = Quantity(meta['ALTITUDE'], 'meter')
    return EarthLocation(lon=lon, lat=lat, height=height)


def _time_ref_from_dict(meta):
    mjd = meta['MJDREFI'] + meta['MJDREFF']
    # TODO: Is 'tt' a default we should put here?
    scale = meta.get('TIMESYS', 'tt').lower()
    # Note: we could call .copy('iso') or .replicate('iso')
    # here if we prefer 'iso' over 'mjd' format in most places.
    return Time(mjd, format='mjd', scale=scale)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.coordinates import Angle, EarthLocation
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity

__all__ = ["earth_location_from_dict", "LazyFitsData"]


class LazyFitsData(object):
    """A lazy FITS data descriptor.

    Parameters
    ----------
    cache : bool
        Whether to cache the data.
    """

    def __init__(self, cache=True):
        self.cache = cache

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, objtype):
        if instance is None:
            # Accessed on a class, not an instance
            return self

        value = instance.__dict__.get(self.name)
        if value is not None:
            return value
        else:
            hdu_loc = instance.__dict__.get(self.name + "_hdu")
            value = hdu_loc.load()
            if self.cache:
                instance.__dict__[self.name] = value
            return value

    def __set__(self, instance, value):
        from gammapy.data import HDULocation

        if isinstance(value, HDULocation):
            instance.__dict__[self.name + "_hdu"] = value
        else:
            instance.__dict__[self.name] = value


# TODO: add unit test
def earth_location_from_dict(meta):
    """Create `~astropy.coordinates.EarthLocation` from FITS header dict."""
    lon = Angle(meta["GEOLON"], "deg")
    lat = Angle(meta["GEOLAT"], "deg")
    # TODO: should we support both here?
    # Check latest spec if ALTITUDE is used somewhere.
    if "GEOALT" in meta:
        height = Quantity(meta["GEOALT"], "meter")
    elif "ALTITUDE" in meta:
        height = Quantity(meta["ALTITUDE"], "meter")
    else:
        raise KeyError("The GEOALT or ALTITUDE header keyword must be set")

    return EarthLocation(lon=lon, lat=lat, height=height)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Latitude, Longitude, Angle
from ..image.core import SkyImage
from ..image import lon_lat_circle_mask

__all__ = [
    'make_tevcat_exclusion_mask'
]


def make_tevcat_exclusion_mask():
    """Create an all-sky exclusion mask containing all TeVCat sources

    Returns
    -------
    mask : `~gammapy.image.SkyImage`
        Exclusion mask
    """
    from gammapy.catalog import load_catalog_tevcat

    tevcat = load_catalog_tevcat()
    all_sky_exclusion = SkyImage.empty(nxpix=3600, nypix=1800, binsz=0.1,
                                       fill=1, dtype='int')
    coords = all_sky_exclusion.coordinates()
    lons = coords.l
    lats = coords.b

    for source in tevcat:
        lon = Longitude(source['coord_gal_lon'], 'deg')
        lat = Latitude(source['coord_gal_lat'], 'deg')
        x = Angle(source['size_x'], 'deg')
        y = Angle(source['size_y'], 'deg')
        if np.isnan(x) and np.isnan(y):
            rad = Angle('0.3 deg')
        else:
            rad = x if x > y else y

        mask = lon_lat_circle_mask(lons, lats, lon, lat, rad)
        all_sky_exclusion.data[mask] = 0
        all_sky_exclusion.meta["EXTNAME"] = "EXCLUSION"

    return all_sky_exclusion

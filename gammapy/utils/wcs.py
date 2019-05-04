# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""WCS related utility functions."""
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import Angle

__all__ = [
    "get_wcs_ctype",
    "get_resampled_wcs",
]


def get_wcs_ctype(wcs):
    """
    Get celestial coordinate type of WCS instance.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        WCS transformation instance.

    Returns
    -------
    ctype : {'galatic', 'icrs'}
        String specifying the coordinate type, that can be used with
        `~astropy.coordinates.SkyCoord`
    """
    ctype = wcs.wcs.ctype
    if "GLON" in ctype[0] or "GLON" in ctype[1]:
        return "galactic"
    elif "RA" in ctype[0] or "RA" in ctype[1]:
        return "icrs"
    else:
        raise TypeError("Can't determine WCS coordinate type.")


def get_resampled_wcs(wcs, factor, downsampled):
    """
    Get resampled WCS object.
    """
    wcs = wcs.deepcopy()

    if not downsampled:
        factor = 1.0 / factor

    wcs.wcs.cdelt *= factor
    wcs.wcs.crpix = (wcs.wcs.crpix - 0.5) / factor + 0.5
    return wcs

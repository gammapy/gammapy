# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [
    'find_reflected_regions',
]


def find_reflected_regions(region, center, exclusion_mask):
    """Find reflected regions.

    Parameters
    ----------
    region : `~gammapy.region.Region`
        Region
    center : `~astropy.coordinates.SkyCoord`
        Rotation point
    exclusion_mask : `~gammapy.region.ExclusionMask`
        TODO

    Returns
    -------
    regions : `~gammapy.region.RegionList`
        Reflected regions list
    """
    raise NotImplementedError

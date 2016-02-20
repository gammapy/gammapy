# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from . import PixRegionList, PixCircleRegion
from astropy.coordinates import Angle

__all__ = [
    'find_reflected_regions',
]


def find_reflected_regions(region, center, exclusion_mask, angle_increment=None,
                           min_distance=None, min_distance_input=None):
    """Find reflected regions.

    Converts to pixel coordinates internally.

    Parameters
    ----------
    region : `~gammapy.region.Region`
        Region
    center : `~astropy.coordinates.SkyCoord`
        Rotation point
    exclusion_mask : `~gammapy.image.ExclusionMask`
        Exlusion mask
    angle_increment : `~astropy.coordinates.Angle`
        Rotation angle for each step
    min_distance : `~astropy.coordinates.Angle`
        Minimal distance between to reflected regions
    min_distance_input : `~astropy.coordinates.Angle`
        Minimal distance from input region

    Returns
    -------
    regions : `~gammapy.region.SkyRegionList`
        Reflected regions list
    """
    angle_increment = Angle('0.1 rad') if angle_increment is None else Angle(angle_increment)
    min_distance = Angle('0 rad') if min_distance is None else Angle(min_distance)
    min_distance_input = Angle('0 rad') if min_distance_input is None else Angle(min_distance_input)

    reflected_regions_pix = PixRegionList()
    wcs = exclusion_mask.wcs
    pix_region = region.to_pixel(wcs)
    val = center.to_pixel(wcs, origin=1)
    pix_center = (float(val[0]), float(val[1]))
    offset = pix_region.offset(pix_center)
    angle = pix_region.angle(pix_center)
    min_ang = Angle(2 * pix_region.radius / offset, 'rad') + min_distance
    max_angle = angle + Angle('360deg') - min_ang - min_distance_input

    curr_angle = angle + min_ang + min_distance_input
    while curr_angle < max_angle:
        test_pos = _compute_xy(pix_center, offset, curr_angle)
        test_reg = PixCircleRegion(test_pos, pix_region.radius)
        if not test_reg.is_inside_exclusion(exclusion_mask):
            reflected_regions_pix.append(test_reg)
            curr_angle = curr_angle + min_ang
        else:
            curr_angle = curr_angle + angle_increment

    reflected_regions = reflected_regions_pix.to_sky(wcs)
    return reflected_regions


def _compute_xy(pix_center, offset, angle):
    """Compute x, y position for a given position angle and offset

    # TODO: replace by calculation using `astropy.coordinates`
    """
    dx = offset * np.sin(angle)
    dy = offset * np.cos(angle)
    x = pix_center[0] + dx
    y = pix_center[1] + dy
    return x, y

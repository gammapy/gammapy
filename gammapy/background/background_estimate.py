# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from .ring import ring_area_factor
from .reflected import find_reflected_regions

__all__ = [
    'BackgroundEstimate',
    'ring_background_estimate',
    'reflected_regions_background_estimate'
]


class BackgroundEstimate(object):
    """Container class for background estimate

    This container holds the result from a region based background estimation
    for one observation. Currently, it is filled by the functions
    :func:`~gammapy.background.ring_background_estimate` and
    :func:`~gammapy.background.reflected_regions_background_estimate`. At some
    points this should be replaced by a more elaborate analysis class.

    Parameters
    ----------
    off_region : `~regions.SkyRegion`
        Background extraction region
    off_events : `~gammapy.data.EventList`
        Background events
    a_on : float
        Relative background exposure of the on region
    a_off : float
        Relative background exposure of the off region
    tag : str
        Background estimation method
    """

    def __init__(self, off_region, off_events, a_on, a_off, tag='default'):
        self.off_region = off_region
        self.off_events = off_events
        self.a_on = a_on
        self.a_off = a_off
        self.tag = tag


def ring_background_estimate(pos, on_radius, inner_radius, outer_radius, events):
    """Simple ring background estimate

    No acceptance correction is applied

    TODO : Replace with AnnulusSkyRegion

    Parameters
    ----------
    pos : `~astropy.coordinates.SkyCoord`
        On region radius
    inner_radius : `~astropy.coordinates.Angle`
        Inner ring radius
    outer_radius : `~astropy.coordinates.Angle`
        Outer ring radius
    events : `~gammapy.data.EventList`
        Events
    """
    off_events = events.select_sky_ring(pos, inner_radius, outer_radius)
    off_region = dict(inner=inner_radius, outer=outer_radius)
    a_on = 1
    a_off = ring_area_factor(on_radius, inner_radius, outer_radius)

    return BackgroundEstimate(off_region, off_events, a_on, a_off, tag='ring')


def reflected_regions_background_estimate(on_region, pointing, exclusion,
                                          events, **kwargs):
    """Reflected regions background estimate

    kwargs are forwaded to :func:`gammapy.background.find_reflected_regions`

    Parameters
    ----------
    on_region : `~regions.CircleSkyRegion`
        On region
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing position
    exclusion : `~gammapy.image.SkyImage`
        Exclusion mask
    events : `gammapy.data.EventList`
        Events
    """
    off_region = find_reflected_regions(on_region, pointing, exclusion, **kwargs)
    off_events = events.select_circular_region(off_region)
    a_on = 1
    a_off = len(off_region)

    return BackgroundEstimate(off_region, off_events, a_on, a_off, tag='reflected')

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from .ring import ring_area_factor

__all__ = [
    'BackgroundEstimate',
    'ring_background_estimate',
]


class BackgroundEstimate(object):
    """Cointainer class for background estimate

    This is meant to hold the result from a region based background estimation
    for one observation. 

    Parameters:
    -----------
    off_events : `~gammapy.data.EventList` 
        Background events 
    off_region : `~astropy.regions.SkyRegion`
        Background estimation region
    alpha : float
        Background scaling factor
    tag : str
        Background estimation method
    """
    def __init__(self, off_region, off_events, alpha, tag='default'):
        self.tag = tag        
        self.off_region = off_region
        self.off_events = off_events
        self.alpha = alpha


def ring_background_estimate(pos, on_radius, inner_radius, outer_radius, events):
    """Simple ring background estimate

    No acceptance correction is applied
    """
    off_events = events.select_sky_ring(pos, inner_radius, outer_radius)
    # TODO : Replace with AnnulusSkyRegion
    off_region = dict(inner=inner_radius, outer=outer_radius)
    alpha = ring_area_factor(on_radius, inner_radius, outer_radius)

    return BackgroundEstimate(off_region, off_events, alpha, tag='ring')


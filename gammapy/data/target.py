# licensed under a 3-clause bsd style license - see license.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from ..stats import Stats
from astropy.table import vstack as table_vstack

__all__ = [
    'Target',
    'TargetSummary',
]

class Target(object):
    """Observation Target.

    This class represents an observation target

    Parameters
    ----------
    position : `~astropy.coordinates.SkyCoord`
        Target position
    on_region : `~astropy.regions.SkyRegion`
        Signal extraction region
    name : str, optional
        Target name
    tag : str, optional
        Target identifier
    """
    def __init__(self, position, on_region, background=None, obs_id=None, 
                 name='Target', tag='target'):
        self.position = position
        self.on_region = on_region
        self.background = background
        self.name = name
        self.tag = tag
        self.obs_id = obs_id

    def add_obs_from_store(self, data_store):
        """Add a list of `~gammapy.data.DataStoreObservations`"""
        self.obs = [data_store.obs(_) for _ in self.obs_id]

    def estimate_background(self, method='ring', **kwargs):
        """Wrapper for different background estimators"""

        if method == 'ring':
            from gammapy.background import ring_background_estimate as ring
            pos = self.position
            on_radius = self.on_region.radius
            inner_radius = kwargs['inner_radius']
            outer_radius = kwargs['outer_radius']
            self.background  = [ring(pos, on_radius, inner_radius, outer_radius,
                                     _.events) for _ in self.obs] 


class TargetSummary(object):
    """Summary Info for an observation Target
    
    Parameters
    ----------
    target : `~gammapy.data.Target`
        Observation target
    obs : list of `~gammapy.data.DataStoreObservation`
        List of observations
    """ 

    def __init__(self, target):
        self.target = target

    @property
    def stats(self):
        """Calculate `~gammapy.stats.Stats`"""
        if self.target.background is None:
            raise ValueError("Need Background estimate for target" )
    
        idx = self.target.on_region.contains(self.events.radec)
        on_events = self.target.events[idx]
        n_on = len(on_events)

        n_off = len(self.background.off_events)
        alpha = self.background.alpha
        return Stats(n_on, n_off, 1, alpha)


    @property
    def events(self):
        """All events"""
        return table_vstack([_.events for _ in self.target.obs])

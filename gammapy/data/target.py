# licensed under a 3-clause bsd style license - see license.rst
from __future__ import absolute_import, division, print_function, unicode_literals

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
    def __init__(self, position, on_region, background=None, name='Target', tag='target'):
        self.position = position
        self.on_region = on_region
        self.background = background
        self.name = name
        self.tag = tag


class TargetSummary(object):
    """Summary Info for an observation Target
    
    Parameters
    ----------
    target : `~gammapy.data.Target`
        Observation target
    obs : list of `~gammapy.data.DataStoreObservation`
        List of observations
    """ 

    def __init__(self, target, obs):
        self.target = target
        self.obs = obs
        self.background = None

    @property
    def stats(self):
        """Calculate `~gammapy.stats.Stats`"""
        if self.background is None:
            raise ValueError("Need Background estimate")
    
        from gammapy.stats import Stats
        idx = self.target.on_region.contains(self.events.radec)
        on_events = self.events[idx]
        n_on = len(on_events)
        n_off = len(self.background.off_events)
        alpha = self.background.alpha
        return Stats(n_on, n_off, 1, alpha)


    def estimate_background(self, method='ring', **kwargs):
        """Wrapper for different background estimators"""

        if method != 'ring':
            raise NotImplementedError
        from gammapy.background import ring_background_estimate
        self.background = ring_background_estimate(self.target, self.events, **kwargs)

    @property
    def events(self):
        """All events"""
        return table_vstack([_.events for _ in self.obs])

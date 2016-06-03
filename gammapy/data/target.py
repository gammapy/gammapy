# licensed under a 3-clause bsd style license - see license.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from ..stats import Stats
from astropy.table import vstack as table_vstack
from astropy.coordinates import SkyCoord
from regions.shapes import CircleSkyRegion
import astropy.units as u

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
    obs_id : int, optional
        Observatinos for this target
    name : str, optional
        Target name
    tag : str, optional
        Target identifier
    """
    def __init__(self, position, on_region, obs_id=None, name='Target', tag='target'):
        self.position = position
        self.on_region = on_region
        self.name = name
        self.tag = tag
        self.obs_id = obs_id
        self.background = None

    def __str__(self):
        ss = "Target: {}\n".format(self.name)
        ss += "Tag: {}\n".format(self.tag)
        ss += "Position: {}\n".format(self.position)
        ss += "On region: {}\n".format(self.on_region)
        return ss
    
    @classmethod
    def from_config(cls, config):
        """Initialize target

        The config dict is stored as attribute for later use by other analysis
        classes
        """
        obs_id = config['obs']
        if not isinstance(obs_id, list):
            from . import ObservationTable
            obs_table = ObservationTable.read(obs_id)
            obs_id = obs_table['OBS_ID'].data
        # TODO : This should also accept also Galactic coordinates
        pos = SkyCoord(config['ra'], config['dec'], unit='deg')
        on_radius = config['on_size'] * u.deg
        on_region = CircleSkyRegion(pos, on_radius)
        target = cls(pos, on_region, obs_id=obs_id,
                     name=config['name'], tag=config['tag'])
        target.config = config
        return target
    
    def add_obs_from_store(self, data_store):
        """Add a list of `~gammapy.data.DataStoreObservations`"""
        if self.obs_id is None:
            raise ValueError("No observations specified")
        self.obs = [data_store.obs(_) for _ in self.obs_id]

    # TODO: This should probably go into a separat BackgroundEstimator class
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
        elif method == 'reflected':
            on_region = self.on_region
            exclusion = kwargs['exclusion']
            from gammapy.background import reflected_regions_background_estimate as refl
            bkg = [refl(on_region, _.pointing_radec, exclusion,
                        _.events) for _ in self.obs]
            self.background = bkg
        else:
            raise NotImplementedError('{}'.format(method))

    def run_spectral_analysis(self, outdir=None):
        """Run spectral analysis

        This runs a spectral analysis with the parameters attached as config
        dict

        Parameters
        ----------
        outdir : Path
            Analysis dir
        """ 
        from . import DataStore
        from ..image import ExclusionMask
        from ..spectrum import SpectrumExtraction

        conf = self.config
        data_store = DataStore.from_all(conf['datastore'])
        self.add_obs_from_store(data_store)
        exclusion = ExclusionMask.read(conf['exclusion_mask']) or None
        conf.update(exclusion=exclusion)
        self.estimate_background(method=conf['background_method'], **conf)
        # Use default energy binning
        self.extraction = SpectrumExtraction(self)
        self.extraction.run(outdir=outdir)

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
        """Total `~gammapy.stats.Stats`"""
        if self.target.background is None:
            raise ValueError("Need Background estimate for target" )
    
        idx = self.target.on_region.contains(self.events.radec)
        on_events = self.events[idx]
        n_on = len(on_events)

        n_off = np.sum([len(_.off_events) for _ in self.target.background])
        # FIXME : This is only true for the ring bg
        alpha = self.target.background[0].alpha
        return Stats(n_on, n_off, 1, alpha)

    @property
    def events(self):
        """All events"""
        return table_vstack([_.events for _ in self.target.obs])

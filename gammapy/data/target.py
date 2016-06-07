# licensed under a 3-clause bsd style license - see license.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from ..stats import Stats
from astropy.table import vstack as table_vstack
from astropy.coordinates import SkyCoord
from ..extern.regions.shapes import CircleSkyRegion
import astropy.units as u

__all__ = [
    'Target',
    'TargetSummary',
]

class Target(object):
    """Observation Target.

    This class represents an observation target. It serves as input for several
    analysis classes, e.g. `~gammapy.spectrum.SpectrumExtraction` and
    `~gammapy.data.TargetSummary`. Some analyses, e.g. background estimation,
    are functions on the ``Target`` class, but should be refactored as separate
    analysis classes. Each analysis class can add attributes to the ``Target``
    instance in order to make some analysis steps, e.g. background estimation
    reusable.

    Parameters
    ----------
    on_region : `~astropy.regions.SkyRegion`
        Signal extraction region
    position : `~astropy.coordinates.SkyCoord`, optional
        Target position
    obs_id : int, optional
        Observatinos for this target
    name : str, optional
        Target name
    tag : str, optional
        Target identifier

    Examples
    --------
    Initialize target and define observations

    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from gammapy.extern.regions.shapes import CircleSkyRegion
    >>> from gammapy.data import Target
    >>> pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame='icrs')
    >>> on_size = 0.3 * u.deg
    >>> on_region = CircleSkyRegion(pos, on_size)
    >>> target = Target(pos, on_region, name='Crab Nebula', tag='crab')
    >>> print(target)
    Target: Crab Nebula
    Tag: crab
    Position: <SkyCoord (ICRS): (ra, dec) in deg
        (83.63, 22.01)>
    On region: CircleSkyRegion
        Center:<SkyCoord (ICRS): (ra, dec) in deg
            (83.63, 22.01)>
        Radius:0.3 deg

    Add data and background estimate

    >>> from gammapy.data import DataStore
    >>> store_dir = "$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2")
    >>> data_store = DataStore.from_dir(store_dir)
    >>> target.obs_id = [23523, 23592]
    >>> target.add_obs_from_store(data_store)
    >>> print(target.obs[0])
    Info for OBS_ID = 23523
    - Start time: 53343.92 s
    - Pointing pos: RA 83.63 deg / Dec 21.51 deg
    - Observation duration: 1687.0 s
    - Dead-time fraction: 6.240 %
    >>> target.estimate_background(method='ring', inner_radius=inner_radius, outer_radius=outer_radius)
    >>> print(len(target.background[0].off_events))
    37

    Create `~gammapy.data.TargetSummary` and `~gammapy.stats.data.Stats`

    >>> summary = target.summary
    >>> type(summary)
    <class 'gammapy.data.target.TargetSummary'>
    >>> stats = summary.stats
    >>> type(stats)
    <class 'gammapy.stats.data.Stats'>

    """
    def __init__(self, on_region, position=None, obs_id=None, name='Target', tag='target'):
        self.on_region = on_region
        self.position = position or on_region.center
        self.obs_id = obs_id
        self.name = name
        self.tag = tag

    def __str__(self):
        """String representation"""
        ss = "Target: {}\n".format(self.name)
        ss += "Tag: {}\n".format(self.tag)
        ss += "On region: {}\n".format(self.on_region)
        return ss
    
    @property
    def summary(self):
        """`~gammapy.data.TargetSummary`"""
        return TargetSummary(self)

    @classmethod
    def from_config(cls, config):
        """Initialize target from config

        The config dict is stored as attribute for later use by other analysis
        classes.
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
            self.background = [ring(pos, on_radius, inner_radius, outer_radius,
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
    
    For an examples see `~gammapy.data.Target`

    Parameters
    ----------
    target : `~gammapy.data.Target`
        Observation target
    """

    def __init__(self, target):
        self.target = target

    @property
    def stats(self):
        """Total `~gammapy.stats.Stats`"""
        if self.target.background is None:
            raise ValueError("Need Background estimate for target")

        idx = self.target.on_region.contains(self.events.radec)
        on_events = self.events[idx]
        n_on = len(on_events)

        n_off = np.sum([len(_.off_events) for _ in self.target.background])
        # FIXME : This is only true for the ring bg
        bkg = self.target.background[0]
        return Stats(n_on, n_off, bkg.a_on, bkg.a_off)

    @property
    def events(self):
        """All events"""
        return table_vstack([_.events for _ in self.target.obs])

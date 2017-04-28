# licensed under a 3-clause bsd style license - see license.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion

__all__ = [
    'Target',
]


class Target(object):
    """Observation Target.

    This class represents an observation target. It can serve as input for several
    analysis classes, e.g. `~gammapy.spectrum.SpectrumExtraction`.

    TODO: This is usefull for pipelines when you want to attach a tag to a
    certain analysis. However, it does not do much. Should we keep it?

    Parameters
    ----------
    on_region : `~regions.SkyRegion`
        Signal extraction region
    position : `~astropy.coordinates.SkyCoord`, optional
        Target position
    obs_id : int, optional
        Observations for this target
    name : str, optional
        Target name
    tag : str, optional
        Target identifier

    Examples
    --------
    Initialize target and define observations:

    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from regions import CircleSkyRegion
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
        ss += "Observations: {}\n".format(len(self.obs_id))
        return ss

    @classmethod
    def from_config(cls, config):
        """Initialize target from config.

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

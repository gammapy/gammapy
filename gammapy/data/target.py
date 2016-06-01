# licensed under a 3-clause bsd style license - see license.rst
from __future__ import absolute_import, division, print_function, unicode_literals


__all__ = [
    'Target',
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
    def __init__(self, position, on_region, name='Target', tag='target'):
        self.position = position
        self.on_region = on_region
        self.name = name
        self.tag = tag

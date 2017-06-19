# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Location of gamma-ray observatories.

TODO: Define observers and convenience functions to convert celestial
SkyCoords to the horizontal altitude-azimuth system.
Or show how to do that in the docs.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.coordinates import Angle, EarthLocation
from ..extern.bunch import Bunch

__all__ = [
    'observatory_locations',
]


def _obs_loc(lon, lat, height):
    return EarthLocation(
        lon=Angle(lon, unit='deg'),
        lat=Angle(lat, unit='deg'),
        height=Quantity(height, 'm'),
    )


# TODO: make this appear in the Sphinx docs!
# https://github.com/gammapy/gammapy/issues/111
# https://github.com/astropy/astropy-helpers/issues/117
"""Cherenkov telescope observatory Earth locations.

Available observatories (alphabetical order):
- CTA (`Website <http://www.cta-observatory.org/>`__, `Wikipedia <http://en.wikipedia.org/wiki/Cherenkov_Telescope_Array>`__):
  Several candidate sites are listed here:
  CTA_South_Aar, CTA_South_Armazones, CTA_South_Leoncito, CTA_North_Teide, CTA_North_Meteor_Crater
- HAWC (`Website <>`__, `Wikipedia <http://en.wikipedia.org/wiki/High_Altitude_Water_Cherenkov_Experiment>`__)
- HEGRA (`Wikipedia <http://en.wikipedia.org/wiki/HEGRA>`__)
- HESS (`Website <https://www.mpi-hd.mpg.de/hfm/HESS/>`__, `Wikipedia <http://en.wikipedia.org/wiki/HESS>`__)
- MAGIC (`Website <https://wwwmagic.mpp.mpg.de/>`__, `Wikipedia <http://en.wikipedia.org/wiki/MAGIC_(telescope)>`__)
- MILAGRO (`Website <>`__, `Wikipedia <http://en.wikipedia.org/wiki/Milagro_(experiment)>`__)
- VERITAS (`Website <http://veritas.sao.arizona.edu/>`__, `Wikipedia <http://en.wikipedia.org/wiki/VERITAS>`__)
- WHIPPLE (`Wikipedia <http://en.wikipedia.org/wiki/Fred_Lawrence_Whipple_Observatory>`__)

Examples
--------

from gammapy.data import observatory_locations
observatory_locations['HESS']
observatory_locations.HESS
list(observatory_locations.keys())
"""
# Locations / height info mostly taken from Wikipedia unless noted differently
observatory_locations = Bunch()
# The selection of CTA candidate sites is somewhat random ... there's other candidate sites I think
# http://en.wikipedia.org/wiki/Cherenkov_Telescope_Array#Site_Selection
observatory_locations['CTA_South_Aar'] = _obs_loc(lon='16.44d', lat='-26.69d', height=1650)
observatory_locations['CTA_South_Armazones'] = _obs_loc(lon='-70.24d', lat='-24.58d', height=2500)
observatory_locations['CTA_South_Armazones_2K'] = _obs_loc(lon='-70.31d', lat='-24.69d', height=2100)
observatory_locations['CTA_South_Leoncito'] = _obs_loc(lon='-69.27d', lat='31.72d', height=2640)
observatory_locations['CTA_North_Teide'] = _obs_loc(lon='-16.54d', lat='28.28d', height=2290)
observatory_locations['CTA_North_Meteor_Crator'] = _obs_loc(lon='-111.03d', lat='35.04d', height=1680)
# HAWC location taken from http://arxiv.org/pdf/1108.6034v2.pdf
observatory_locations['HAWC'] = _obs_loc(lon='-97d18m34s', lat='18d59m48s', height=4100)
observatory_locations['HEGRA'] = _obs_loc(lon='28d45m42s', lat='17d53m27s', height=3)
# Precision position of HESS from the HESS software (slightly different from Wikipedia)
observatory_locations['HESS'] = _obs_loc(lon='16d30m00.8s', lat='-23d16m18.4s', height=1835)
observatory_locations['MAGIC'] = _obs_loc(lon='-17d53m24s', lat='28d45m43s', height=2200)
observatory_locations['MILAGRO'] = _obs_loc(lon='-106.67625d', lat='35.87835d', height=2530)
observatory_locations['VERITAS'] = _obs_loc(lon='-110d57m07.77s', lat='31d40m30.21s', height=1268)
# WHIPPLE coordinates taken from the Observatory Wikipedia page:
# http://en.wikipedia.org/wiki/Fred_Lawrence_Whipple_Observatory
observatory_locations['WHIPPLE'] = _obs_loc(lon='-110d52m42s', lat='31d40m52s', height=2606)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sky maps."""
from .axes import MapAxes, MapAxis, TimeMapAxis, LabelMapAxis
from .coord import MapCoord
from .core import Map
from .geom import Geom
from .hpx import HpxMap, HpxGeom, HpxNDMap
from .maps import Maps
from .region import RegionGeom, RegionNDMap
from .wcs import WcsMap, WcsGeom, WcsNDMap


__all__ = [
    # axes
    "MapAxes",
    "MapAxis",
    "TimeMapAxis",
    "LabelMapAxis",
    # coord
    "MapCoord",
    # core
    "Map",
    # geom
    "Geom",
    # hpx
    "HpxMap",
    "HpxGeom",
    "HpxNDMap",
    # maps
    "Maps",
    # region
    "RegionGeom",
    "RegionNDMap",
    # wcs
    "WcsMap",
    "WcsGeom",
    "WcsNDMap",
]

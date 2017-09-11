# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from astropy.coordinates import SkyCoord
from ..base import MapBase
from ..geom import MapAxis

pytest.importorskip('scipy')
pytest.importorskip('healpy')

map_axes = [
    MapAxis.from_bounds(1.0, 10.0, 3, interp='log'),
    MapAxis.from_bounds(0.1, 1.0, 4, interp='log'),
]

mapbase_args = [
    (0.1, 10.0, 'wcs', SkyCoord(0.0, 30.0, unit='deg'), None),
    (0.1, 10.0, 'wcs', SkyCoord(0.0, 30.0, unit='deg'), map_axes[:1]),
    (0.1, 10.0, 'wcs', SkyCoord(0.0, 30.0, unit='deg'), map_axes),
    (0.1, 10.0, 'hpx', SkyCoord(0.0, 30.0, unit='deg'), None),
    (0.1, 10.0, 'hpx', SkyCoord(0.0, 30.0, unit='deg'), map_axes[:1]),
    (0.1, 10.0, 'hpx', SkyCoord(0.0, 30.0, unit='deg'), map_axes),
    (0.1, 10.0, 'hpx-sparse', SkyCoord(0.0, 30.0, unit='deg'), None),
]


@pytest.mark.parametrize(('binsz', 'width', 'map_type', 'skydir', 'axes'),
                         mapbase_args)
def test_mapbase_create(binsz, width, map_type, skydir, axes):
    m = MapBase.create(binsz=binsz, width=width, map_type=map_type,
                       skydir=skydir, axes=axes)

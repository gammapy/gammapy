# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from astropy.coordinates import SkyCoord
from collections import OrderedDict
from ..base import Map
from ..geom import MapAxis
from ..wcs import WcsGeom
from ..wcsnd import WcsNDMap
from ..hpx import HpxGeom
from ..hpxnd import HpxNDMap


pytest.importorskip('scipy')
pytest.importorskip('healpy')
pytest.importorskip('numpy', '1.12.0')


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
def test_map_create(binsz, width, map_type, skydir, axes):
    m = Map.create(binsz=binsz, width=width, map_type=map_type,
                   skydir=skydir, axes=axes)


def test_map_from_geom():
    geom = WcsGeom.create(binsz=1.0, width=10.0)
    m = Map.from_geom(geom)
    assert isinstance(m, WcsNDMap)

    geom = HpxGeom.create(binsz=1.0, width=10.0)
    m = Map.from_geom(geom)
    assert isinstance(m, HpxNDMap)


@pytest.mark.parametrize('map_type', ['wcs', 'hpx', 'hpx-sparse'])
def test_map_meta_read_write(map_type):
    meta = OrderedDict([
        ('user', 'test'),
    ])

    m = Map.create(binsz=0.1, width=10.0, map_type=map_type,
                   skydir=SkyCoord(0.0, 30.0, unit='deg'), meta=meta)

    hdulist = m.to_hdulist(hdu='COUNTS')
    header = hdulist['COUNTS'].header

    assert header['META'] == '{"user": "test"}'

    m2 = Map.from_hdu_list(hdulist)
    assert m2.meta == meta

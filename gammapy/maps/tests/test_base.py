# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from collections import OrderedDict
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Unit, Quantity
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
    (0.1, 10.0, 'wcs', SkyCoord(0.0, 30.0, unit='deg'), None, ''),
    (0.1, 10.0, 'wcs', SkyCoord(0.0, 30.0, unit='deg'), map_axes[:1], ''),
    (0.1, 10.0, 'wcs', SkyCoord(0.0, 30.0, unit='deg'), map_axes, 'm^2'),
    (0.1, 10.0, 'hpx', SkyCoord(0.0, 30.0, unit='deg'), None, ''),
    (0.1, 10.0, 'hpx', SkyCoord(0.0, 30.0, unit='deg'), map_axes[:1], ''),
    (0.1, 10.0, 'hpx', SkyCoord(0.0, 30.0, unit='deg'), map_axes, 's^2'),
    (0.1, 10.0, 'hpx-sparse', SkyCoord(0.0, 30.0, unit='deg'), None, ''),
]


@pytest.mark.parametrize(('binsz', 'width', 'map_type', 'skydir', 'axes', 'unit'),
                         mapbase_args)
def test_map_create(binsz, width, map_type, skydir, axes, unit):
    m = Map.create(binsz=binsz, width=width, map_type=map_type,
                   skydir=skydir, axes=axes, unit=unit)
    assert m.unit == unit


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


unit_args = [
    ('wcs', 's'),
    ('wcs', ''),
    ('wcs', Unit('sr')),
    ('hpx', 'm^2')
]


@pytest.mark.parametrize(('map_type', 'unit'), unit_args)
def test_map_quantity(map_type, unit):
    m = Map.create(binsz=0.1, width=10.0, map_type=map_type, unit=unit)

    # This is to test if default constructor with no unit performs as expected
    if unit is None:
        unit = ''
    assert m.quantity.unit == Unit(unit)

    m.quantity = Quantity(np.ones_like(m.data), 'm2')
    assert m.unit == 'm2'


@pytest.mark.parametrize(('map_type', 'unit'), unit_args)
def test_map_unit_read_write(map_type, unit):
    m = Map.create(binsz=0.1, width=10.0, map_type=map_type, unit=unit)

    hdu_list = m.to_hdulist(hdu='COUNTS')
    header = hdu_list['COUNTS'].header

    assert Unit(header['UNIT']) == Unit(unit)

    m2 = Map.from_hdu_list(hdu_list)
    assert m2.unit == unit


@pytest.mark.parametrize(('map_type', 'unit'), unit_args)
def test_map_repr(map_type, unit):
    m = Map.create(binsz=0.1, width=10.0, map_type=map_type, unit=unit)
    assert unit in repr(m)
    assert m.__class__.__name__ in repr(m)

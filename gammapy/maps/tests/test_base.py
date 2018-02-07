# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from astropy.coordinates import SkyCoord
from collections import OrderedDict
from ..base import Map
from ..geom import MapAxis

pytest.importorskip('scipy')
pytest.importorskip('healpy')


def make_test_map(map_type, meta):
    """
    Make empty maps

    Parameters
    ----------
    map_type: str
        Define wcs or hpx
    meta : `~collections.OrderedDict`
        Dictionary to store meta data.

    Returns
    -------
    m: `~gammapy.maps.Map`

    """
    map_axes = [
        MapAxis.from_bounds(1.0, 10.0, 3, interp='log'),
        MapAxis.from_bounds(0.1, 1.0, 4, interp='log'),
    ]
    m = Map.create(binsz=0.1, width=10.0, map_type=map_type,
                   skydir=SkyCoord(0.0, 30.0, unit='deg'), axes=map_axes, meta=meta)
    return m


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
    m = Map.create(binsz=binsz, width=width, map_type=map_type,
                   skydir=skydir, axes=axes)


def test_write():
    meta = OrderedDict()
    meta["name"] = "counts"
    m_wcs = make_test_map(map_type="wcs", meta=meta)
    m_hpx = make_test_map(map_type="hpx", meta=meta)
    m_hpx_sparse = make_test_map(map_type="hpx-sparse", meta=meta)
    hdulist_wcs = m_wcs.to_hdulist()
    hdulist_hpx = m_hpx.to_hdulist()
    hdulist_hpx_sparse = m_hpx_sparse.to_hdulist()
    header_wcs = hdulist_wcs[0].header
    header_hpx = hdulist_hpx[0].header
    header_hpx_sparse = hdulist_hpx_sparse[0].header
    assert "name" in header_wcs
    assert "name" in header_hpx
    assert "name" in header_hpx_sparse
    


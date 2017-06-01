# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ..geom import MapAxis
from ..hpx import HPXGeom
from ..hpxcube import HpxCube

pytest.importorskip('healpy')


hpx_test_geoms = [(8, False, 'GAL', None, None),
                  (8, False, 'GAL', None, [MapAxis(np.logspace(0., 3., 4))]),
                  (8, False, 'GAL', 'DISK(110.,75.,10.)',
                   [MapAxis(np.logspace(0., 3., 4))]),
                  (8, False, 'GAL', 'DISK(110.,75.,10.)',
                   [MapAxis(np.logspace(0., 3., 4), name='axis0'),
                    MapAxis(np.logspace(0., 2., 3), name='axis1')])
                  ]


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_init(nside, nested, coordsys, region, axes):

    geom = HPXGeom(nside, nested, coordsys, region=region, axes=axes)
    shape = [np.unique(geom.npix)]
    if axes:
        shape += [ax.nbin for ax in axes]
    shape = shape[::-1]
    data = np.random.uniform(0, 1, shape)
    m = HpxCube(geom)
    assert(m.data.shape == data.shape)
    m = HpxCube(geom, data)
    assert_allclose(m.data, data)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_read_write(tmpdir, nside, nested, coordsys, region, axes):

    filename = str(tmpdir / 'skycube.fits')
    m = HpxCube(HPXGeom(nside, nested, coordsys, region=region, axes=axes))
    data = np.random.poisson(0.1, m.data.shape)
    m.data[...] = data
    m.write(filename)
    m2 = HpxCube.read(filename)
    assert_allclose(m.data, m2.data)
    m.write(filename, sparse=True)
    m2 = HpxCube.read(filename)
    assert_allclose(m.data, m2.data)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_get_by_pix(nside, nested, coordsys, region, axes):
    m = HpxCube(HPXGeom(nside, nested, coordsys, region=region, axes=axes))
    data = np.linspace(0, m.data.size-1.0,m.data.size).reshape(m.data.shape)
    m.data[...] = data
    pix = m.hpx.get_pixels()
    assert_allclose(np.ravel(m.data), m.get_by_pix(pix))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_get_by_coord(nside, nested, coordsys, region, axes):
    m = HpxCube(HPXGeom(nside, nested, coordsys, region=region, axes=axes))
    data = np.linspace(0, m.data.size-1.0,m.data.size).reshape(m.data.shape)
    m.data[...] = data
    coords = m.hpx.get_coords()
    assert_allclose(np.ravel(m.data), m.get_by_coord(coords))


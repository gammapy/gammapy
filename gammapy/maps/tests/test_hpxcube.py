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


def make_test_cubes():

    data = np.random.uniform(0, 1, 3072)
    m0 = HpxCube(HPXGeom(16, False, 'GAL'), data)
    m1 = HpxCube(HPXGeom(16, False, 'GAL', axes=[np.linspace(0., 3., 4)]))
    m2 = HpxCube(HPXGeom(16, False, 'GAL', region='DISK(110.,75.,10.)',
                         axes=[np.linspace(0., 3., 4)]))
    m3 = HpxCube(HPXGeom(16, False, 'GAL', region='DISK(110.,75.,10.)',
                         axes=[np.linspace(0., 3., 4), np.linspace(0., 2., 3)]))
    return m0, m1, m2, m3


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         [(8, False, 'GAL', None, None),
                          (8, False, 'GAL', None, [
                           MapAxis(np.logspace(0., 3., 4))]),
                          (8, False, 'GAL', 'DISK(110.,75.,10.)',
                           [MapAxis(np.logspace(0., 3., 4))]),
                          (8, False, 'GAL', 'DISK(110.,75.,10.)',
                           [MapAxis(np.logspace(0., 3., 4), name='axis0'),
                            MapAxis(np.logspace(0., 2., 3))])
                          ])
def test_hpxcube_init(nside, nested, coordsys, region, axes):

    shape = []
    if axes:
        shape = [ax.nbin for ax in axes]
    geom = HPXGeom(nside, nested, coordsys, region=region, axes=axes)
    shape += [np.unique(geom.npix)]
    data = np.random.uniform(0, 1, shape)
    m = HpxCube(geom)
    assert(m.data.shape == data.shape)
    m = HpxCube(geom, data)
    assert_allclose(m.data, data)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         [(8, False, 'GAL', None, None),
                          (8, False, 'GAL', None, [
                           MapAxis(np.logspace(0., 3., 4))]),
                          (8, False, 'GAL', 'DISK(110.,75.,10.)',
                           [MapAxis(np.logspace(0., 3., 4))]),
                          (8, False, 'GAL', 'DISK(110.,75.,10.)',
                           [MapAxis(np.logspace(0., 3., 4), name='axis0'),
                            MapAxis(np.logspace(0., 2., 3))])
                          ])
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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..geom import MapAxis
from ..hpx import HpxGeom
from ..hpxcube import HpxMapND

pytest.importorskip('scipy')
pytest.importorskip('healpy')

hpx_test_geoms = [
    (8, False, 'GAL', None, None),
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
    geom = HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes)
    shape = [int(np.unique(geom.npix))]
    if axes:
        shape += [ax.nbin for ax in axes]
    shape = shape[::-1]
    data = np.random.uniform(0, 1, shape)
    m = HpxMapND(geom)
    assert m.data.shape == data.shape
    m = HpxMapND(geom, data)
    assert_allclose(m.data, data)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_create(nside, nested, coordsys, region, axes):
    m = HpxMapND.create(nside=nside, nest=nested, coordsys=coordsys,
                        region=region, axes=axes)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_read_write(tmpdir, nside, nested, coordsys, region, axes):
    filename = str(tmpdir / 'skycube.fits')
    m = HpxMapND(HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes))
    data = np.random.poisson(0.1, m.data.shape)
    m.data[...] = data
    m.write(filename)
    m2 = HpxMapND.read(filename)
    assert_allclose(m.data, m2.data)
    m.write(filename, sparse=True)
    m2 = HpxMapND.read(filename)
    assert_allclose(m.data, m2.data)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_get_by_pix(nside, nested, coordsys, region, axes):
    m = HpxMapND(HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes))
    data = np.linspace(0, m.data.size - 1.0, m.data.size).reshape(m.data.shape)
    m.data[...] = data
    pix = m.hpx.get_pixels()
    assert_allclose(np.ravel(m.data), m.get_by_pix(pix))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_get_by_coords(nside, nested, coordsys, region, axes):
    m = HpxMapND(HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes))
    data = np.linspace(0, m.data.size - 1.0, m.data.size).reshape(m.data.shape)
    m.data[...] = data
    coords = m.hpx.get_coords()
    assert_allclose(np.ravel(m.data), m.get_by_coords(coords))


@pytest.mark.xfail(reason="Bug in healpy <= 0.10.3")
@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_get_by_coords_interp(nside, nested, coordsys, region, axes):
    m = HpxMapND(HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes))
    coords = m.hpx.get_coords()
    m.data[...] = coords[1].reshape(m.data.shape)
    assert_allclose(np.ravel(m.data), m.get_by_coords(coords, interp='linear'))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_fill_by_coords(nside, nested, coordsys, region, axes):
    m = HpxMapND(HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes))
    coords = m.hpx.get_coords()
    m.fill_by_coords(coords, coords[1])
    m.fill_by_coords(coords, coords[1])
    assert_allclose(np.ravel(m.data), 2.0 * coords[1])


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_set_by_coords(nside, nested, coordsys, region, axes):
    m = HpxMapND(HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes))
    coords = m.hpx.get_coords()
    m.set_by_coords(coords, coords[1])
    assert_allclose(np.ravel(m.data), coords[1])


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_to_wcs(nside, nested, coordsys, region, axes):
    m = HpxMapND(HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes))
    m_wcs = m.to_wcs(sum_bands=False, oversample=2, normalize=False)
    m_wcs = m.to_wcs(sum_bands=True, oversample=2, normalize=False)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_swap_scheme(nside, nested, coordsys, region, axes):
    m = HpxMapND(HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes))
    m.data[...] = np.random.poisson(1.0, m.data.shape)
    m2 = m.to_swapped_scheme()
    coords = m.hpx.get_coords()
    assert_allclose(m.get_by_coords(coords), m2.get_by_coords(coords))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxcube_ud_grade(nside, nested, coordsys, region, axes):
    m = HpxMapND(HpxGeom(nside=nside, nest=nested, coordsys=coordsys, region=region, axes=axes))
    m.to_ud_graded(m.hpx.order - 1)

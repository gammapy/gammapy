# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..utils import fill_poisson
from ..geom import MapAxis
from ..base import Map
from ..hpx import HpxGeom
from ..hpxmap import HpxMap
from ..hpxnd import HpxNDMap
from ..hpxsparse import HpxSparseMap

pytest.importorskip('scipy')
pytest.importorskip('healpy')
pytest.importorskip('numpy', '1.12.0')

axes1 = [MapAxis(np.logspace(0., 3., 3), interp='log')]

hpx_test_allsky_geoms = [
    (8, False, 'GAL', None, None),
    (8, False, 'GAL', None, axes1),
    ([4, 8], False, 'GAL', None, axes1)]

hpx_test_partialsky_geoms = [
    ([4, 8], False, 'GAL', 'DISK(110.,75.,30.)', axes1),
    (8, False, 'GAL', 'DISK(110.,75.,10.)',
     [MapAxis(np.logspace(0., 3., 4))]),
    (8, False, 'GAL', 'DISK(110.,75.,10.)',
     [MapAxis(np.logspace(0., 3., 4), name='axis0'),
      MapAxis(np.logspace(0., 2., 3), name='axis1')])
]

hpx_test_geoms = hpx_test_allsky_geoms + hpx_test_partialsky_geoms

hpx_test_geoms_sparse = [tuple(list(t) + [True]) for t in hpx_test_geoms]
hpx_test_geoms_sparse += [tuple(list(t) + [False]) for t in hpx_test_geoms]


def create_map(nside, nested, coordsys, region, axes, sparse):
    if sparse:
        m = HpxSparseMap(HpxGeom(nside=nside, nest=nested,
                                 coordsys=coordsys, region=region, axes=axes))
    else:
        m = HpxNDMap(HpxGeom(nside=nside, nest=nested,
                             coordsys=coordsys, region=region, axes=axes))

    return m


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxmap_init(nside, nested, coordsys, region, axes):
    geom = HpxGeom(nside=nside, nest=nested,
                   coordsys=coordsys, region=region, axes=axes)
    shape = [int(np.max(geom.npix))]
    if axes:
        shape += [ax.nbin for ax in axes]
    shape = shape[::-1]
    data = np.random.uniform(0, 1, shape)
    m = HpxNDMap(geom)
    assert m.data.shape == data.shape
    m = HpxNDMap(geom, data)
    assert_allclose(m.data, data)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes', 'sparse'),
                         hpx_test_geoms_sparse)
def test_hpxmap_create(nside, nested, coordsys, region, axes, sparse):
    m = create_map(nside, nested, coordsys, region, axes, sparse)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes', 'sparse'),
                         hpx_test_geoms_sparse)
def test_hpxmap_read_write(tmpdir, nside, nested, coordsys, region, axes, sparse):
    filename = str(tmpdir / 'skycube.fits')

    m = create_map(nside, nested, coordsys, region, axes, sparse)
    fill_poisson(m, mu=0.5, random_state=0)
    m.write(filename)

    m2 = HpxNDMap.read(filename)
    m3 = HpxSparseMap.read(filename)
    m4 = Map.read(filename, map_type='hpx')
    if sparse:
        msk = np.isfinite(m2.data[...])
    else:
        msk = np.ones_like(m2.data[...], dtype=bool)

    assert_allclose(m.data[...][msk], m2.data[...][msk])
    assert_allclose(m.data[...][msk], m3.data[...][msk])
    assert_allclose(m.data[...][msk], m4.data[...][msk])

    m.write(filename, sparse=True)
    m2 = HpxNDMap.read(filename)
    m3 = HpxMap.read(filename, map_type='hpx')
    m4 = Map.read(filename, map_type='hpx')
    assert_allclose(m.data[...][msk], m2.data[...][msk])
    assert_allclose(m.data[...][msk], m3.data[...][msk])
    assert_allclose(m.data[...][msk], m4.data[...][msk])


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes', 'sparse'),
                         hpx_test_geoms_sparse)
def test_hpxmap_set_get_by_pix(nside, nested, coordsys, region, axes, sparse):
    m = create_map(nside, nested, coordsys, region, axes, sparse)
    coords = m.geom.get_coords(flat=True)
    idx = m.geom.get_idx(flat=True)
    m.set_by_pix(idx, coords[0])
    assert_allclose(coords[0], m.get_by_pix(idx))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes', 'sparse'),
                         hpx_test_geoms_sparse)
def test_hpxmap_set_get_by_coords(nside, nested, coordsys, region, axes, sparse):
    m = create_map(nside, nested, coordsys, region, axes, sparse)
    coords = m.geom.get_coords(flat=True)
    m.set_by_coords(coords, coords[0])
    assert_allclose(coords[0], m.get_by_coords(coords))


@pytest.mark.xfail(reason="Bug in healpy <= 0.10.3")
@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxmap_get_by_coords_interp(nside, nested, coordsys, region, axes):
    m = HpxNDMap(HpxGeom(nside=nside, nest=nested,
                         coordsys=coordsys, region=region, axes=axes))
    coords = m.geom.get_coords()
    m.set_by_coords(coords, coords[1])
    assert_allclose(m.get_by_coords(coords),
                    m.get_by_coords(coords, interp='linear'))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes', 'sparse'),
                         hpx_test_geoms_sparse)
def test_hpxmap_fill_by_coords(nside, nested, coordsys, region, axes, sparse):
    m = create_map(nside, nested, coordsys, region, axes, sparse)
    coords = m.geom.get_coords(flat=True)
    m.fill_by_coords(coords, coords[1])
    m.fill_by_coords(coords, coords[1])
    assert_allclose(m.get_by_coords(coords), 2.0 * coords[1])


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxmap_iter(nside, nested, coordsys, region, axes):
    m = HpxNDMap(HpxGeom(nside=nside, nest=nested,
                         coordsys=coordsys, region=region, axes=axes))
    coords = m.geom.get_coords(flat=True)
    m.fill_by_coords(coords, coords[0])
    for vals, pix in m.iter_by_pix(buffersize=100):
        assert_allclose(vals, m.get_by_pix(pix))
    for vals, coords in m.iter_by_coords(buffersize=100):
        assert_allclose(vals, m.get_by_coords(coords))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxmap_to_wcs(nside, nested, coordsys, region, axes):
    m = HpxNDMap(HpxGeom(nside=nside, nest=nested,
                         coordsys=coordsys, region=region, axes=axes))
    m_wcs = m.to_wcs(sum_bands=False, oversample=2, normalize=False)
    m_wcs = m.to_wcs(sum_bands=True, oversample=2, normalize=False)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxmap_swap_scheme(nside, nested, coordsys, region, axes):
    m = HpxNDMap(HpxGeom(nside=nside, nest=nested,
                         coordsys=coordsys, region=region, axes=axes))
    fill_poisson(m, mu=1.0, random_state=0)
    m2 = m.to_swapped_scheme()
    coords = m.geom.get_coords(flat=True)
    assert_allclose(m.get_by_coords(coords), m2.get_by_coords(coords))


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxmap_ud_grade(nside, nested, coordsys, region, axes):
    m = HpxNDMap(HpxGeom(nside=nside, nest=nested,
                         coordsys=coordsys, region=region, axes=axes))
    m.to_ud_graded(4)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_partialsky_geoms)
def test_hpxmap_pad(nside, nested, coordsys, region, axes):
    m = HpxNDMap(HpxGeom(nside=nside, nest=nested,
                         coordsys=coordsys, region=region, axes=axes))
    m.pad(1)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_partialsky_geoms)
def test_hpxmap_crop(nside, nested, coordsys, region, axes):
    m = HpxNDMap(HpxGeom(nside=nside, nest=nested,
                         coordsys=coordsys, region=region, axes=axes))
    m.crop(1)


@pytest.mark.parametrize(('nside', 'nested', 'coordsys', 'region', 'axes'),
                         hpx_test_geoms)
def test_hpxmap_sum_over_axes(nside, nested, coordsys, region, axes):
    m = HpxNDMap(HpxGeom(nside=nside, nest=nested,
                         coordsys=coordsys, region=region, axes=axes))
    coords = m.geom.get_coords(flat=True)
    m.fill_by_coords(coords, coords[0])
    msum = m.sum_over_axes()

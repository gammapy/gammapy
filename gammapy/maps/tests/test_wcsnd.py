# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ..utils import fill_poisson
from ..geom import MapAxis, coordsys_to_frame
from ..base import Map
from ..wcs import WcsGeom
from ..hpx import HpxGeom
from ..wcsmap import WcsMap
from ..wcsnd import WcsNDMap

pytest.importorskip('scipy')
pytest.importorskip('reproject')

axes1 = [MapAxis(np.logspace(0., 3., 3), interp='log')]
axes2 = [MapAxis(np.logspace(0., 3., 3), interp='log'),
         MapAxis(np.logspace(1., 3., 4), interp='lin')]
skydir = SkyCoord(110., 75.0, unit='deg', frame='icrs')

wcs_allsky_test_geoms = [
    (None, 10.0, 'GAL', 'AIT', skydir, None),
    (None, 10.0, 'GAL', 'AIT', skydir, axes1),
    (None, [10.0, 20.0], 'GAL', 'AIT', skydir, axes1),
    (None, 10.0, 'GAL', 'AIT', skydir, axes2),
    (None, [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]],
     'GAL', 'AIT', skydir, axes2),
]

wcs_partialsky_test_geoms = [
    (10, 1.0, 'GAL', 'AIT', skydir, None),
    (10, 1.0, 'GAL', 'AIT', skydir, axes1),
    (10, [1.0, 2.0], 'GAL', 'AIT', skydir, axes1),
    (10, 1.0, 'GAL', 'AIT', skydir, axes2),
    (10, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
     'GAL', 'AIT', skydir, axes2),
]

wcs_test_geoms = wcs_allsky_test_geoms + wcs_partialsky_test_geoms


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_init(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m0 = WcsNDMap(geom)
    coords = m0.geom.get_coords()
    m0.set_by_coords(coords, coords[1])
    m1 = WcsNDMap(geom, m0.data)
    assert_allclose(m0.data, m1.data)


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_read_write(tmpdir, npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    filename = str(tmpdir / 'skycube.fits')
    filename_sparse = str(tmpdir / 'skycube_sparse.fits')
    m0 = WcsNDMap(geom)
    fill_poisson(m0, mu=0.5)
    m0.write(filename)
    m1 = WcsNDMap.read(filename)
    m2 = Map.read(filename)
    m3 = Map.read(filename, map_type='wcs')
    assert_allclose(m0.data, m1.data)
    assert_allclose(m0.data, m2.data)
    assert_allclose(m0.data, m3.data)

    m0.write(filename_sparse, sparse=True)
    m1 = WcsNDMap.read(filename_sparse)
    m2 = Map.read(filename)
    m3 = Map.read(filename, map_type='wcs')
    assert_allclose(m0.data, m1.data)
    assert_allclose(m0.data, m2.data)
    assert_allclose(m0.data, m3.data)


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_set_get_by_pix(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, skydir=skydir,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    coords = m.geom.get_coords()
    pix = m.geom.get_idx()
    m.set_by_pix(pix, coords[0])
    assert_allclose(coords[0], m.get_by_pix(pix))


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_set_get_by_coords(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, skydir=skydir,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    coords = m.geom.get_coords()
    m.set_by_coords(coords, coords[0])
    assert_allclose(coords[0], m.get_by_coords(coords))

    if not geom.is_allsky:
        coords[1][...] = 0.0
        assert_allclose(
            np.nan * np.ones(coords[0].shape), m.get_by_coords(coords))

    # Test with SkyCoords
    m = WcsNDMap(geom)
    coords = m.geom.get_coords()
    skydir = SkyCoord(coords[0], coords[1], unit='deg',
                      frame=coordsys_to_frame(geom.coordsys))
    skydir_cel = skydir.transform_to('icrs')
    skydir_gal = skydir.transform_to('galactic')
    m.set_by_coords((skydir_gal,) + coords[2:], coords[0])
    assert_allclose(coords[0], m.get_by_coords(coords))
    assert_allclose(m.get_by_coords((skydir_cel,) + coords[2:]),
                    m.get_by_coords((skydir_gal,) + coords[2:]))


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_fill_by_coords(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, skydir=skydir,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    coords = m.geom.get_coords()
    fill_coords = tuple([np.concatenate((t, t)) for t in coords])
    fill_vals = fill_coords[1]
    m.fill_by_coords(fill_coords, fill_vals)
    assert_allclose(m.get_by_coords(coords), 2.0 * coords[1])

    # Test with SkyCoords
    m = WcsNDMap(geom)
    coords = m.geom.get_coords()
    skydir = SkyCoord(coords[0], coords[1], unit='deg',
                      frame=coordsys_to_frame(geom.coordsys))
    skydir_cel = skydir.transform_to('icrs')
    skydir_gal = skydir.transform_to('galactic')
    fill_coords_cel = (skydir_cel,) + coords[2:]
    fill_coords_gal = (skydir_gal,) + coords[2:]
    m.fill_by_coords(fill_coords_cel, coords[1])
    m.fill_by_coords(fill_coords_gal, coords[1])
    assert_allclose(m.get_by_coords(coords), 2.0 * coords[1])


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_coadd(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, skydir=skydir,
                          proj=proj, coordsys=coordsys, axes=axes)
    m0 = WcsNDMap(geom)
    m1 = WcsNDMap(geom.upsample(2))
    coords = m0.geom.get_coords()
    m1.fill_by_coords(tuple([np.concatenate((t, t)) for t in coords]),
                      np.concatenate((coords[1], coords[1])))
    m0.coadd(m1)
    assert_allclose(np.nansum(m0.data), np.nansum(m1.data), rtol=1E-4)


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_interp_by_coords(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, skydir=skydir,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    coords = m.geom.get_coords(flat=True)
    m.set_by_coords(coords, coords[1])
    assert_allclose(coords[1], m.interp_by_coords(coords, interp='nearest'))
    assert_allclose(coords[1], m.interp_by_coords(coords, interp='linear'))
    assert_allclose(coords[1], m.interp_by_coords(coords, interp=1))
    if geom.is_regular and not geom.is_allsky:
        assert_allclose(coords[1], m.interp_by_coords(coords, interp='cubic'))


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_iter(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    coords = m.geom.get_coords()
    m.fill_by_coords(coords, coords[0])
    for vals, pix in m.iter_by_pix(buffersize=100):
        assert_allclose(vals, m.get_by_pix(pix))
    for vals, coords in m.iter_by_coords(buffersize=100):
        assert_allclose(vals, m.get_by_coords(coords))


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_sum_over_axes(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    coords = m.geom.get_coords()
    m.fill_by_coords(coords, coords[0])
    msum = m.sum_over_axes()


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_reproject(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj,
                          skydir=skydir, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)

    if geom.projection == 'AIT' and geom.is_allsky:
        pytest.xfail('Bug in reproject version <= 0.3.1')

    if geom.ndim > 3 or geom.npix[0].size > 1:
        pytest.xfail(
            "> 3 dimensions or multi-resolution geometries not supported")

    geom0 = WcsGeom.create(npix=npix, binsz=binsz, proj=proj,
                           skydir=skydir, coordsys=coordsys, axes=axes)
    m0 = m.reproject(geom0, order=1)

    assert_allclose(m.data, m0.data)

    # TODO : Reproject to a different spatial geometry


def test_wcsndmap_reproject_allsky_car():
    geom = WcsGeom.create(binsz=10.0, proj='CAR', coordsys='CEL')
    m = WcsNDMap(geom)
    coords = m.geom.get_coords()
    m.set_by_coords(coords, coords[0])

    geom0 = WcsGeom.create(binsz=1.0, proj='CAR', coordsys='CEL',
                           skydir=(180.0, 0.0), width=30.0)
    m0 = m.reproject(geom0, order=1)
    coords0 = m0.geom.get_coords()
    assert_allclose(m0.get_by_coords(coords0), coords0[0])

    geom1 = HpxGeom.create(binsz=5.0, coordsys='CEL')
    m1 = m.reproject(geom1, order=1)
    coords1 = m1.geom.get_coords()

    m = (coords1[0] > 10.0) & (coords1[0] < 350.0)
    assert_allclose(m1.get_by_coords((coords1[0][m], coords1[1][m])),
                    coords1[0][m])


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_pad(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    m2 = m.pad(1, mode='constant', cval=2.2)
    if not geom.is_allsky:
        coords = m2.geom.get_coords()
        msk = m2.geom.contains(coords)
        coords = tuple([c[~msk] for c in coords])
        assert_allclose(m2.get_by_coords(coords), 2.2)
    m.pad(1, mode='edge')
    m.pad(1, mode='interp')


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_crop(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    m.crop(1)


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_downsample(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    # Check whether we can downsample
    if (np.all(np.mod(geom.npix[0], 2) == 0) and
            np.all(np.mod(geom.npix[1], 2) == 0)):
        m2 = m.downsample(2, preserve_counts=True)
        assert_allclose(np.nansum(m.data), np.nansum(m2.data))


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsndmap_upsample(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsNDMap(geom)
    m2 = m.upsample(2, order=0, preserve_counts=True)
    assert_allclose(np.nansum(m.data), np.nansum(m2.data))

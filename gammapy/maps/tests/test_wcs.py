# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ..wcs import WcsGeom
from ..geom import MapAxis

pytest.importorskip('scipy')

axes1 = [MapAxis(np.logspace(0., 3., 3), interp='log', name='energy')]
axes2 = [MapAxis(np.logspace(0., 3., 3), interp='log', name='energy'),
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
    (10, 0.1, 'GAL', 'AIT', skydir, None),
    (10, 0.1, 'GAL', 'AIT', skydir, axes1),
    (10, [0.1, 0.2], 'GAL', 'AIT', skydir, axes1),
]

wcs_test_geoms = wcs_allsky_test_geoms + wcs_partialsky_test_geoms


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsgeom_init(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, skydir=skydir,
                          proj=proj, coordsys=coordsys, axes=axes)


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsgeom_get_pix(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, skydir=skydir,
                          proj=proj, coordsys=coordsys, axes=axes)
    pix = geom.get_idx()
    if axes is not None:
        idx = tuple([1] * len(axes))
        pix_img = geom.get_idx(idx=idx)
        m = np.all(np.stack([x == y for x, y in zip(idx, pix[2:])]), axis=0)
        m2 = pix_img[0] != -1
        assert_allclose(pix[0][m], np.ravel(pix_img[0][m2]))
        assert_allclose(pix[1][m], np.ravel(pix_img[1][m2]))


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsgeom_test_pix_to_coord(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, skydir=skydir,
                          proj=proj, coordsys=coordsys, axes=axes)
    assert_allclose(geom.get_coord()[0],
                    geom.pix_to_coord(geom.get_idx())[0])


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsgeom_test_coord_to_idx(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    assert_allclose(geom.get_idx()[0],
                    geom.coord_to_idx(geom.get_coord())[0])

    if not geom.is_allsky:
        coords = geom.center_coord[:2] + \
            tuple([ax.center[0] for ax in geom.axes])
        coords[0][...] += 2.0 * np.max(geom.width[0])
        idx = geom.coord_to_idx(coords)
        assert_allclose(np.full_like(coords[0], -1, dtype=int), idx[0])
        idx = geom.coord_to_idx(coords, clip=True)
        assert np.all(np.not_equal(np.full_like(coords[0], -1, dtype=int), idx[0]))


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsgeom_read_write(tmpdir, npix, binsz, coordsys, proj, skydir, axes):
    geom0 = WcsGeom.create(npix=npix, binsz=binsz,
                           proj=proj, coordsys=coordsys, axes=axes)

    shape = (np.max(geom0.npix[0]), np.max(geom0.npix[1]))
    hdu_bands = geom0.make_bands_hdu(hdu='BANDS')
    hdu_prim = fits.PrimaryHDU(np.zeros(shape).T)
    hdu_prim.header.update(geom0.make_header())

    filename = str(tmpdir / 'wcsgeom.fits')
    hdulist = fits.HDUList([hdu_prim, hdu_bands])
    hdulist.writeto(filename, overwrite=True)

    hdulist = fits.open(filename)
    geom1 = WcsGeom.from_header(hdulist[0].header, hdulist['BANDS'])

    assert_allclose(geom0.npix, geom1.npix)
    assert (geom0.coordsys == geom1.coordsys)


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsgeom_contains(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, skydir=skydir,
                          proj=proj, coordsys=coordsys, axes=axes)
    coords = geom.get_coord()
    coords = [c[np.isfinite(c)] for c in coords]
    assert_allclose(geom.contains(coords),
                    np.ones(coords[0].shape, dtype=bool))

    if axes is not None:
        coords = [c[0] for c in coords[:2]] + \
                 [ax.edges[-1] + 1.0 for ax in axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))

    if not geom.is_allsky:
        coords = [0.0, 0.0] + [ax.center[0] for ax in geom.axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))

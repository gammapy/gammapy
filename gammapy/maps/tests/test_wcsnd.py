# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ..geom import MapAxis
from ..wcs import WcsGeom
from ..hpx import HpxGeom
from ..wcsnd import WcsMapND


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
def test_wcsmapnd_init(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m0 = WcsMapND(geom)
    m0.fill_poisson(0.5)
    m1 = WcsMapND(geom, m0.data)
    assert_allclose(m0.data, m1.data)


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsmapnd_read_write(tmpdir, npix, binsz, coordsys, proj, skydir, axes):

    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    filename = str(tmpdir / 'skycube.fits')
    filename_sparse = str(tmpdir / 'skycube_sparse.fits')
    m0 = WcsMapND(geom)
    m0.fill_poisson(0.5)
    m0.write(filename)
    m1 = WcsMapND.read(filename)
    assert_allclose(m0.data, m1.data)
    m0.write(filename_sparse, sparse=True)
    m1 = WcsMapND.read(filename_sparse)
    assert_allclose(m0.data, m1.data)


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsmapnd_fill_by_coords(tmpdir, npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsMapND(geom)
    coords = m.geom.get_coords()
    m.fill_by_coords(tuple([np.concatenate((t, t)) for t in coords]),
                     np.concatenate((coords[1], coords[1])))
    assert_allclose(m.get_by_coords(coords), 2.0 * coords[1])


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsmapnd_iter(tmpdir, npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsMapND(geom)
    coords = m.geom.get_coords()
    m.fill_by_coords(coords, coords[0])
    for vals, pix in m.iter_by_pix(buffersize=100):
        assert_allclose(vals, m.get_by_pix(pix))
    for vals, coords in m.iter_by_coords(buffersize=100):
        assert_allclose(vals, m.get_by_coords(coords))


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsmapnd_sum_over_axes(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz,
                          proj=proj, coordsys=coordsys, axes=axes)
    m = WcsMapND(geom)
    coords = m.geom.get_coords()
    m.fill_by_coords(coords, coords[0])
    msum = m.sum_over_axes()


@pytest.mark.parametrize(('npix', 'binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsmapnd_reproject(npix, binsz, coordsys, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj,
                          skydir=skydir, coordsys=coordsys, axes=axes)
    m = WcsMapND(geom)

    if geom.projection == 'AIT' and geom.allsky:
        pytest.xfail('Bug in reproject version <= 0.3.1')

    if geom.ndim > 3 or geom.npix[0].size > 1:
        pytest.xfail(
            "> 3 dimensions or multi-resolution geometries not supported")

    geom0 = WcsGeom.create(npix=npix, binsz=binsz, proj=proj,
                           skydir=skydir, coordsys=coordsys, axes=axes)
    m0 = m.reproject(geom0, order=1)

    assert_allclose(m.data, m0.data)

    # TODO : Reproject to a different spatial geometry


def test_wcsmapnd_reproject_allsky_car():

    geom = WcsGeom.create(binsz=10.0, proj='CAR', coordsys='CEL')
    m = WcsMapND(geom)
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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ..geom import MapAxis
from ..wcs import WcsGeom
from ..wcsnd import WcsMapND


pytest.importorskip('scipy')

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
    m0 = WcsMapND(geom)
    m0.fill_poisson(0.5)
    m0.write(filename)
    m1 = WcsMapND.read(filename)
    assert_allclose(m0.data, m1.data)
    m0.write(filename, sparse=True)
    m1 = WcsMapND.read(filename)
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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ..wcs import WCSGeom
from ..geom import MapAxis

wcs_test_geoms = [
    (10.0, 'GAL', 'AIT', SkyCoord(110., 75.0, unit='deg', frame='icrs'), None),
    (10.0, 'GAL', 'AIT', SkyCoord(110., 75.0, unit='deg', frame='icrs'),
     [MapAxis(np.logspace(0., 3., 4), interp='log')]),
    (10.0, 'GAL', 'AIT', SkyCoord(110., 75.0, unit='deg', frame='icrs'),
     [MapAxis(np.logspace(0., 3., 4), interp='log'),
      MapAxis(np.logspace(1., 3., 3), interp='lin')]),
]


@pytest.mark.parametrize(('binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsgeom_init(binsz, coordsys, proj, skydir, axes):
    geom = WCSGeom.create(binsz=binsz, proj=proj, coordsys=coordsys, axes=axes)


@pytest.mark.parametrize(('binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsgeom_test_pix_to_coord(binsz, coordsys, proj, skydir, axes):
    geom = WCSGeom.create(binsz=binsz, proj=proj, coordsys=coordsys, axes=axes)
    geom.pix_to_coord(geom.get_pixels())


@pytest.mark.parametrize(('binsz', 'coordsys', 'proj', 'skydir', 'axes'),
                         wcs_test_geoms)
def test_wcsgeom_test_coord_to_pix(binsz, coordsys, proj, skydir, axes):
    geom = WCSGeom.create(binsz=binsz, proj=proj, coordsys=coordsys, axes=axes)
    geom.coord_to_pix(geom.get_coords())

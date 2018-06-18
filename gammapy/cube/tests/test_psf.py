# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
import astropy.units as u
from ...maps import MapAxis, WcsGeom, Map
from ..psf import WcsNDMapPSFKernel


@pytest.fixture(scope='session')
def geom():
    axis = MapAxis.from_edges(np.logspace(-1, 1, 3), unit=u.TeV)
    return WcsGeom.create(skydir=(0, 0), npix=(5, 4), coordsys='GAL', axes=[axis])


@pytest.fixture(scope='session')
def exposure(geom):
    m = Map.from_geom(geom)
    m.quantity = np.ones((2, 4, 5)) * u.Quantity('100 m2 s')
    return m


@pytest.fixture(scope='session')
def psf(geom):
    return WcsNDMapPSFKernel.from_gauss(geom, sigma='0.3 deg')


def test_psf_kernel_create(psf):
    assert psf.data

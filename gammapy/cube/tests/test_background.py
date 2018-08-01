# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.coordinates import SkyCoord, Angle
from ...utils.testing import requires_data
from ...maps import Map
from ...irf import Background3D
from ..background import make_map_background_irf

pytest.importorskip('scipy')


@pytest.fixture(scope='session')
def bkg_3d():
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    return Background3D.read(filename, hdu='BACKGROUND')


@pytest.fixture(scope='session')
def counts_cube():
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/hess_events_simulated_023523_cntcube.fits'
    return Map.read(filename)


@requires_data('gammapy-extra')
def test_make_map_fov_background(bkg_3d, counts_cube):
    pointing = SkyCoord(83.633, 21.514, unit='deg')
    livetime = Quantity(1581.17, 's')

    m = make_map_background_irf(
        pointing, livetime, bkg_3d, counts_cube.geom
    )

    assert m.data.shape == (15, 120, 200)
    assert_allclose(m.data[0, 0, 0], 0.013959, rtol=1e-4)
    assert_allclose(m.data.sum(), 1408.573698, rtol=1e-5)

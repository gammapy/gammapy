# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.coordinates import SkyCoord, Angle
from ...utils.testing import requires_data
from ...maps import WcsNDMap
from ...irf import Background3D
from ..background import make_map_background_irf

pytest.importorskip('scipy')


@pytest.fixture(scope='session')
def bkg_3d():
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    return Background3D.read(filename, hdu='BACKGROUND')


@pytest.fixture(scope='session')
def counts_cube():
    import os
    filename = os.path.join(
        os.environ['GAMMAPY_EXTRA'],
        'datasets/hess-crab4-hd-hap-prod2/hess_events_simulated_023523_cntcube.fits'
    )
    return WcsNDMap.read(filename)


@requires_data('gammapy-extra')
def test_make_map_fov_background(bkg_3d, counts_cube):
    pointing = SkyCoord(83.633, 21.514, unit='deg')
    livetime = Quantity(1581.17, 's')
    offset_max = Angle(2.2, 'deg')

    m = make_map_background_irf(
        pointing, livetime, bkg_3d, counts_cube.geom, offset_max,
    )

    assert m.data.shape == (15, 120, 200)
    assert_allclose(m.data[0, 0, 0], 0.013959, rtol=1e-4)
    assert_allclose(m.data.sum(), 1356.2551, rtol=1e-5)

    # TODO: Check that `offset_max` is working properly
    # pos = SkyCoord(85.6, 23, unit='deg')
    # val = bkg_cube.lookup(pos, energy=1 * u.TeV)
    # assert_allclose(val, 0)

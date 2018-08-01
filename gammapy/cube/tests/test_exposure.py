# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from astropy.coordinates import SkyCoord, Angle
from ...utils.testing import requires_data, assert_quantity_allclose
from ...maps import Map
from ...irf import EffectiveAreaTable2D
from ..exposure import make_map_exposure_true_energy

pytest.importorskip('scipy')


@pytest.fixture(scope='session')
def aeff():
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz'
    return EffectiveAreaTable2D.read(filename, hdu='AEFF_2D')


@pytest.fixture(scope='session')
def counts_cube():
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/hess_events_simulated_023523_cntcube.fits'
    return Map.read(filename)


@requires_data('gammapy-extra')
def test_make_map_exposure_true_energy(aeff, counts_cube):

    m = make_map_exposure_true_energy(
        pointing=SkyCoord(83.633, 21.514, unit='deg'),
        livetime='1581.17 s',
        aeff=aeff,
        geom=counts_cube.geom,
        offset_max=Angle('2.2 deg'),
    )

    assert m.data.shape == (15, 120, 200)
    assert m.unit == 'm2 s'
    assert_quantity_allclose(np.nanmax(m.data), 4.7e8, rtol=100)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_data
from ...maps import WcsGeom, HpxGeom, MapAxis
from ...irf import EffectiveAreaTable2D
from ..exposure import make_map_exposure_true_energy

pytest.importorskip('scipy')
pytest.importorskip('healpy')


@pytest.fixture(scope='session')
def aeff():
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz'
    return EffectiveAreaTable2D.read(filename, hdu='AEFF_2D')


@pytest.fixture(scope='session')
def geom():
    axis = MapAxis.from_edges([0.1, 1, 10], name="energy", unit='TeV')
    return WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis])


@requires_data('gammapy-extra')
def test_make_map_exposure_true_energy(aeff, geom):
    m = make_map_exposure_true_energy(
        pointing=SkyCoord(2, 1, unit='deg'),
        livetime='1581.17 s',
        aeff=aeff,
        geom=geom,
    )

    assert m.data.shape == (2, 3, 4)
    assert m.unit == 'm2 s'
    assert_allclose(m.data.sum(), 3.215819e+09, rtol=1e-5)

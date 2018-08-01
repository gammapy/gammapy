# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_data
from ...data import DataStore
from ...maps import WcsGeom, MapAxis
from ..make import MapMaker

pytest.importorskip('scipy')


@pytest.fixture(scope='session')
def obs_list():
    data_store = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/")
    obs_id = [110380, 111140]
    return data_store.obs_list(obs_id)


@pytest.fixture(scope='session')
def geom():
    skydir = SkyCoord(266.41681663, -29.00782497, unit="deg")
    energy_axis = MapAxis.from_edges([0.1, 0.5, 1.5, 3.0, 10.],
                                     name='energy', unit='TeV', interp='log')
    return WcsGeom.create(binsz=0.1 * u.deg, skydir=skydir, width=15.0, axes=[energy_axis])


@requires_data('gammapy-extra')
@pytest.mark.parametrize("pars", [
    {
        'mode': 'trim',
        'counts': 107214,
        'exposure': 9.582158e+13,
        'background': 107214.016,
    },
    {
        'mode': 'strict',
        'counts': 53486,
        'exposure': 4.794064e+13,
        'background': 53486,
    },
])
def test_map_maker(pars, obs_list, geom):
    maker = MapMaker(geom, '6 deg', cutout_mode=pars['mode'])
    maps = maker.run(obs_list)

    counts = maps['counts_map']
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), pars['counts'], rtol=1e-5)

    exposure = maps['exposure_map']
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.sum(), pars['exposure'], rtol=1e-5)

    background = maps['background_map']
    assert background.unit == ""
    assert_allclose(background.data.sum(), pars['background'], rtol=1e-5)

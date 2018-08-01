# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from ...utils.testing import assert_quantity_allclose, requires_data
from ...data import DataStore
from ...maps import WcsGeom, MapAxis
from ..make import MapMaker

pytest.importorskip('scipy')


@pytest.fixture(scope='session')
def data_store():
    return DataStore.from_dir("$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/")


@pytest.fixture(scope='session')
def geom():
    skydir = SkyCoord(266.41681663, -29.00782497, unit="deg")
    energy_axis = MapAxis.from_edges([0.1, 0.5, 1.5, 3.0, 10.],
                                     name='energy', unit='TeV', interp='log')
    return WcsGeom.create(binsz=0.1 * u.deg, skydir=skydir, width=15.0, axes=[energy_axis])


@requires_data('gammapy-extra')
@pytest.mark.parametrize("mode, expected", [("trim", 107214.0), ("strict", 53486.0)])
def test_map_maker(mode, expected, data_store, geom):
    mmaker = MapMaker(geom, '6 deg', cutout_mode=mode)
    obs = [110380, 111140]

    for obsid in obs:
        mmaker.process_obs(data_store.obs(obsid))

    assert mmaker.exposure_map.unit == "m2 s"
    assert_quantity_allclose(mmaker.counts_map.data.sum(), expected)

    maker = MapMaker(geom, '6 deg', cutout_mode=mode)
    obslist = data_store.obs_list(obs)
    maps = maker.run(obslist)

    assert maps['exposure_map'].unit == "m2 s"
    assert_quantity_allclose(maps['counts_map'].data.sum(), expected)

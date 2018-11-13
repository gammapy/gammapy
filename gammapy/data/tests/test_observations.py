# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.time import Time
from ...data import DataStore, EventList, GTI, ObservationCTA
from ...irf import EffectiveAreaTable2D, EnergyDispersion2D, PSF3D
from ...utils.testing import requires_data
from ...utils.testing import (
    assert_quantity_allclose,
    assert_time_allclose,
    assert_skycoord_allclose,
)


@pytest.fixture(scope="session")
def data_store():
    return DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/")


@requires_data("gammapy-extra")
def test_data_store_observation(data_store):
    """Test DataStoreObservation class"""
    obs = data_store.obs(23523)

    assert_time_allclose(obs.tstart, Time(53343.92234009259, scale="tt", format="mjd"))
    assert_time_allclose(obs.tstop, Time(53343.94186555556, scale="tt", format="mjd"))

    c = SkyCoord(83.63333129882812, 21.51444435119629, unit="deg")
    assert_skycoord_allclose(obs.pointing_radec, c)

    c = SkyCoord(22.481705, 41.38979, unit="deg")
    assert_skycoord_allclose(obs.pointing_altaz, c)

    c = SkyCoord(83.63333129882812, 22.01444435119629, unit="deg")
    assert_skycoord_allclose(obs.target_radec, c)


@requires_data("gammapy-extra")
def test_create_missing_gti():
    """Test the DataStoreObservation._create_missing_gti() method.

    For the 'hess-dl3-dr1' data set, the GTI tables contain only the 'TSTART' and 'TSTOP' from the 'event list' and
    'obs index' header. The `DataStoreObservation._create_missing_gti()` method creates a GTI table on-the-fly using
    exactly those values.
    """
    ds = data_store()
    obs_time_1 = ds.obs(20136).observation_time_duration

    ds = data_store()  # DataStore saves some lazyproperties when accessing a hdu location -> we create a new one
    ds.hdu_table.remove_row(1)  # Remove reference to GTI table -> GTI table will be created on-the-fly
    obs_time_2 = ds.obs(20136).observation_time_duration

    assert obs_time_1 == obs_time_2


@requires_data("gammapy-extra")
def test_data_store_observation_to_observation_cta(data_store):
    """Test the DataStoreObservation.to_observation_cta conversion method"""
    obs = data_store.obs(23523).to_observation_cta()

    assert type(obs) == ObservationCTA
    assert type(obs.obs_id) == int
    assert type(obs.gti) == GTI
    assert type(obs.events) == EventList
    assert type(obs.aeff) == EffectiveAreaTable2D
    assert type(obs.edisp) == EnergyDispersion2D
    assert type(obs.psf) == PSF3D
    assert type(obs.pointing_radec) == SkyCoord
    assert type(obs.observation_live_time_duration) == Quantity
    assert type(obs.observation_dead_time_fraction) == np.float64


@requires_data("gammapy-extra")
class TestObservationChecker:
    def setup(self):
        data_store = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps")
        self.observation = data_store.obs(111140)

    def test_check_all(self):
        records = list(self.observation.check())
        assert len(records) == 8

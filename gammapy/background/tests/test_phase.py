# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ...data import DataStore, EventList
from .. import BackgroundEstimate, PhaseBackgroundEstimator

@pytest.fixture
def on_region():
    """Example on_region for testing."""
    pos = SkyCoord('08h35m20.65525s', '-45d10m35.1545s', frame='icrs')
    radius = Angle(0.2, 'deg')
    region = CircleSkyRegion(pos, radius)
    return region


@pytest.fixture
def obs_list():
    """Example observation list for testing."""
    DATA_DIR = '$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps'
    datastore = DataStore.from_dir(DATA_DIR)
    obs_ids = [111630]
    return datastore.obs_list(obs_ids)


@pytest.fixture(scope='session')
def phase_bkg_estimator():
    """Example background estimator for testing."""
    estimator = PhaseBackgroundEstimator(obs_list=obs_list(),
                                         on_region=on_region(),
                                         on_phase=(0.5, 0.6),
                                         off_phase=(0.7, 1))
    return estimator


def test_basic(phase_bkg_estimator):
    assert 'PhaseBackgroundEstimator' in str(phase_bkg_estimator)


def test_run(phase_bkg_estimator):
    phase_bkg_estimator.run()
    assert len(phase_bkg_estimator.result) == 1


def test_filter_events(obs_list, on_region):
    all_events = obs_list[0].events.select_circular_region(on_region)
    ev1 = PhaseBackgroundEstimator.filter_events(all_events, (0, 0.3))
    assert isinstance(ev1, EventList)
    ev2 = PhaseBackgroundEstimator.filter_events(all_events, (0.3, 1))
    assert len(all_events.table) == len(ev1.table) + len(ev2.table)


def test_process(phase_bkg_estimator, obs_list):
    assert isinstance(phase_bkg_estimator.process(obs_list[0]), BackgroundEstimate)


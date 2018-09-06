# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ...utils.testing import requires_data
from ...data import DataStore, EventList
from .. import BackgroundEstimate, PhaseBackgroundEstimator


@pytest.fixture
def on_region():
    """Example on_region for testing."""
    pos = SkyCoord("08h35m20.65525s", "-45d10m35.1545s", frame="icrs")
    radius = Angle(0.2, "deg")
    return CircleSkyRegion(pos, radius)


@pytest.fixture
def obs_list():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps")
    return datastore.obs_list([111630])


@pytest.fixture(scope="session")
def phase_bkg_estimator():
    """Example background estimator for testing."""
    return PhaseBackgroundEstimator(
        obs_list=obs_list(),
        on_region=on_region(),
        on_phase=(0.5, 0.6),
        off_phase=(0.7, 1),
    )


@requires_data("gammapy-extra")
def test_basic(phase_bkg_estimator):
    assert "PhaseBackgroundEstimator" in str(phase_bkg_estimator)


@requires_data("gammapy-extra")
def test_run(phase_bkg_estimator):
    phase_bkg_estimator.run()
    assert len(phase_bkg_estimator.result) == 1


@requires_data("gammapy-extra")
def test_filter_events(obs_list, on_region):
    all_events = obs_list[0].events.select_circular_region(on_region)
    ev1 = PhaseBackgroundEstimator.filter_events(all_events, (0, 0.3))
    assert isinstance(ev1, EventList)
    ev2 = PhaseBackgroundEstimator.filter_events(all_events, (0.3, 1))
    assert len(all_events.table) == len(ev1.table) + len(ev2.table)


@pytest.mark.parametrize(
    "pars",
    [
        {"p_in": [[0.2, 0.3]], "p_out": [[0.2, 0.3]]},
        {"p_in": [[0.9, 0.1]], "p_out": [[0.9, 1], [0, 0.1]]},
    ],
)
def test_check_phase_intervals(pars):
    assert PhaseBackgroundEstimator._check_intervals(pars["p_in"]) == pars["p_out"]


@requires_data("gammapy-extra")
def test_process(phase_bkg_estimator, obs_list):
    assert isinstance(phase_bkg_estimator.process(obs_list[0]), BackgroundEstimate)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from astropy.coordinates import SkyCoord, Angle
from ...utils.regions import SphericalCircleSkyRegion
from ...utils.testing import requires_data
from ...data import DataStore, EventList
from .. import BackgroundEstimate, PhaseBackgroundEstimator


@pytest.fixture(scope="session")
def on_region():
    """Example on_region for testing."""
    pos = SkyCoord("08h35m20.65525s", "-45d10m35.1545s", frame="icrs")
    radius = Angle(0.2, "deg")
    return SphericalCircleSkyRegion(pos, radius)


@pytest.fixture(scope="session")
def observations():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")
    return datastore.get_observations([111630])


@pytest.fixture(scope="session")
def phase_bkg_estimator(on_region, observations):
    """Example background estimator for testing."""
    return PhaseBackgroundEstimator(
        observations=observations,
        on_region=on_region,
        on_phase=(0.5, 0.6),
        off_phase=(0.7, 1),
    )


@requires_data()
def test_basic(phase_bkg_estimator):
    assert "PhaseBackgroundEstimator" in str(phase_bkg_estimator)


@requires_data()
def test_run(phase_bkg_estimator):
    phase_bkg_estimator.run()
    assert len(phase_bkg_estimator.result) == 1


@requires_data()
def test_filter_events(observations, on_region):
    all_events = observations[0].events.select_region(on_region)
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


@requires_data()
def test_process(phase_bkg_estimator, observations):
    assert isinstance(phase_bkg_estimator.process(observations[0]), BackgroundEstimate)

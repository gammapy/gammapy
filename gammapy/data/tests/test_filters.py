# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
from astropy import units as u
from regions import CircleSkyRegion
from ...data import DataStore, ObservationFilter, EventListBase
from ...utils.testing import requires_data


def test_event_filter_types():
    for method_str in ObservationFilter.EVENT_FILTER_TYPES.values():
        assert hasattr(EventListBase, method_str)


@pytest.fixture(scope="session")
def observation():
    ds = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/")
    return ds.obs(20136)


@requires_data("gammapy-extra")
def test_empty_observation_filter(observation):
    empty_obs_filter = ObservationFilter()

    events = observation.events
    filtered_events = empty_obs_filter.filter_events(events)
    assert filtered_events == events

    gti = observation.gti
    filtered_gti = empty_obs_filter.filter_gti(gti)
    assert filtered_gti == gti


@requires_data("gammapy-extra")
def test_filter_events(observation):
    custom_filter = {
        "type": "custom",
        "opts": {"parameter": "ENERGY", "limits": (0.8, 10.0)},
    }

    target_position = SkyCoord(ra=229.2, dec=-58.3, unit="deg", frame="icrs")
    region_radius = Angle("0.2 deg")
    region = CircleSkyRegion(center=target_position, radius=region_radius)
    circular_region_filter = {"type": "circular_region", "opts": {"region": region}}

    time_filter = Time([53090.12, 53090.13], format="mjd", scale="tt")

    obs_filter = ObservationFilter(
        event_filters=[custom_filter, circular_region_filter], time_filter=time_filter
    )

    events = observation.events
    filtered_events = obs_filter.filter_events(events)

    assert all(
        (filtered_events.energy >= 0.8 * u.TeV)
        & (filtered_events.energy < 10.0 * u.TeV)
    )
    assert all(
        (filtered_events.time >= time_filter[0])
        & (filtered_events.time < time_filter[1])
    )
    assert all(region.center.separation(filtered_events.radec) < region_radius)


@requires_data("gammapy-extra")
def test_filter_gti(observation):
    time_filter = Time([53090.12, 53090.13], format="mjd", scale="tt")

    obs_filter = ObservationFilter(time_filter=time_filter)

    gti = observation.gti
    filtered_gti = obs_filter.filter_gti(gti)

    assert all(filtered_gti.time_start >= time_filter[0])
    assert all(filtered_gti.time_stop <= time_filter[1])

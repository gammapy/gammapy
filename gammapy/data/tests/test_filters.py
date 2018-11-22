# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...data import DataStore, ObservationFilter, EventListBase
from ...utils.testing import requires_data


def test_event_filter_types():
    for method_str in ObservationFilter.EVENT_FILTER_TYPES.values():
        assert hasattr(EventListBase, method_str)


@requires_data("gammapy-extra")
def test_empty_ObservationFilter():
    ds = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-dl3-dr1")
    obs = ds.obs(20136)

    empty_obs_filter = ObservationFilter()

    events = obs.events
    filtered_events = empty_obs_filter.filter_events(events)
    assert filtered_events == events

    gti = obs.gti
    filtered_gti = empty_obs_filter.filter_gti(gti)
    assert filtered_gti == gti

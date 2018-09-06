# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from .background_estimate import BackgroundEstimate
from gammapy.data import EventList
import numpy as np

__all__ = ["PhaseBackgroundEstimator"]


class PhaseBackgroundEstimator(object):
    """Background estimation with on and off phases.

    This class is responsible for creating a
    `~gammapy.background.BackgroundEstimate` by counting events
    in the on-phase-zone and off-phase-zone in an ON-region,
    given an observation, an on_region, an on-phase-zone, an off-phase-zone.

    For a usage example see future notebook.

    TODO: The phase interval has to be between 0 and 1.
    Cases like [-0.1, 0.1], for example, are still not supported.

    Parameters
    ----------
    on_region : `~regions.CircleSkyRegion`
        Target region in the sky
    obs_list : `~gammapy.data.ObservationList`
        Observations to process
    on_phase : `tuple` or list of tuples
        on-phase defined by the two edges of each interval (edges are excluded)
    off_phase : `tuple` or list of tuples
        off-phase defined by the two edges of each interval (edges are excluded)
    """

    def __init__(self, on_region, on_phase, off_phase, obs_list):
        self.on_region = on_region
        self.obs_list = obs_list
        self.on_phase = np.atleast_2d(on_phase)
        self.off_phase = np.atleast_2d(off_phase)
        self.result = None

    def __str__(self):
        s = self.__class__.__name__
        s += "\n{}".format(self.on_region)
        s += "\n{}".format(self.on_phase)
        s += "\n{}".format(self.off_phase)
        s += "\n{}".format(self.obs_list)
        return s

    def run(self):
        """Run all steps."""
        result = []
        for obs in self.obs_list:
            temp = self.process(obs=obs)
            result.append(temp)

        self.result = result

    @staticmethod
    def filter_events(events, tuple_phase_zone):
        """Select events depending on their phases."""
        p = events.table["PHASE"]
        mask = (tuple_phase_zone[0] < p) & (p < tuple_phase_zone[1])
        return events.select_row_subset(mask)

    @staticmethod
    def _check_intervals(intervals):
        """Split phase intervals that go beyond phase 1"""
        for phase_interval in intervals:
            if phase_interval[0] > phase_interval[1]:
                intervals.remove(phase_interval)
                intervals.append([phase_interval[0], 1])
                intervals.append([0, phase_interval[1]])
        return intervals

    def process(self, obs):
        """Estimate background for one observation."""
        all_events = obs.events.select_circular_region(self.on_region)

        self.on_phase = self._check_intervals(self.on_phase)
        self.off_phase = self._check_intervals(self.off_phase)

        # Loop over all ON- and OFF- phase intervals to filter the ON- and OFF- events
        list_on_events = [
            self.filter_events(all_events, each_on_phase)
            for each_on_phase in self.on_phase
        ]
        list_off_events = [
            self.filter_events(all_events, each_off_phase)
            for each_off_phase in self.off_phase
        ]

        # Loop over all ON- and OFF- phase intervals to compute the normalization factors a_on and a_off
        a_on = np.sum([_[1] - _[0] for _ in self.on_phase])
        a_off = np.sum([_[1] - _[0] for _ in self.off_phase])

        on_events = EventList.stack(list_on_events)
        off_events = EventList.stack(list_off_events)

        return BackgroundEstimate(
            on_region=self.on_region,
            on_events=on_events,
            off_region=None,
            off_events=off_events,
            a_on=a_on,
            a_off=a_off,
            method="Phase Bkg Estimator",
        )

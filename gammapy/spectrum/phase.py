import numpy as np
from gammapy.background.background_estimate import BackgroundEstimate


class PhaseBackgroundEstimator(object):
    """Background estimation with on and off phases.

    This class is responsible for creating a
    `~gammapy.background.BackgroundEstimate` by counting events in the on-phase-zone and off-phase-zone in an ON-region,
    given an observation, an on_region, an on-phase-zone, an off-phase-zone.

    For a usage example see future notebook

    Parameters
    ----------
    on_region : `~regions.CircleSkyRegion`
        Target region
    obs_list : `~gammapy.data.ObservationList`
        Observations to process
    on_phase : `~tuple`
        on-phase-zone defined by a tuple of the two edges of the interval
    off_phase : `~tuple`
        off-phase-zone defined by a tuple of the two edges of the interval
    """

    def __init__(self, on_region, on_phase, off_phase, obs_list):
        self.on_region = on_region
        self.obs_list = obs_list
        self.on_phase = on_phase
        self.off_phase = off_phase

        self.result = None

    def __str__(self):
        s = self.__class__.__name__
        s += '\n{}'.format(self.on_region)
        s += '\n{}'.format(self.on_phase)
        s += '\n{}'.format(self.off_phase)
        s += '\n{}'.format(self.obs_list)
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

    def process(self, obs):
        """Estimate background for one observation."""
        all_events = obs.events.select_circular_region(self.on_region)

        on_events = self.filter_events(all_events, self.on_phase)
        off_events = self.filter_events(all_events, self.off_phase)

        a_on = self.on_phase[1] - self.on_phase[0]
        a_off = self.off_phase[1] - self.off_phase[0]
        return BackgroundEstimate(
            on_region=self.on_region,
            on_events=on_events,
            off_region=None,
            off_events=off_events,
            a_on=a_on,
            a_off=a_off,
            method='Phase Bkg Estimator',
        )

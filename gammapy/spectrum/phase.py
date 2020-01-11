# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.data import EventList
from .core import CountsSpectrum


__all__ = ["PhaseBackgroundMaker"]


class PhaseBackgroundMaker:
    """Background estimation with on and off phases.

    TODO: For a usage example see future notebook.

    TODO: The phase interval has to be between 0 and 1.
    Cases like [-0.1, 0.1], for example, are still not supported.

    Parameters
    ----------
    on_phase : `tuple` or list of tuples
        on-phase defined by the two edges of each interval (edges are excluded)
    off_phase : `tuple` or list of tuples
        off-phase defined by the two edges of each interval (edges are excluded)
    """

    def __init__(self, on_phase, off_phase):
        self.on_phase = self._check_intervals(on_phase)
        self.off_phase = self._check_intervals(off_phase)

    def __str__(self):
        s = self.__class__.__name__
        s += f"\n{self.on_phase}"
        s += f"\n{self.off_phase}"
        return s

    @staticmethod
    def _make_counts(dataset, observation, phases):
        events = observation.events.select_region(dataset.counts.region)

        event_lists = []
        for interval in phases:
            events = events.select_parameter(parameter="PHASE", band=interval)
            event_lists.append(events)

        events_off = EventList.stack(event_lists)

        edges = dataset.counts.energy.edges
        counts = CountsSpectrum(energy_hi=edges[1:], energy_lo=edges[:-1])
        counts.fill_events(events_off)
        return counts

    def make_counts_off(self, dataset, observation):
        """Make off counts.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Input dataset.
        observation : `DatastoreObservation`
            Data store observation.

        Returns
        -------
        counts_off : `CountsSpectrum`
            Off counts.
        """
        return self._make_counts(dataset, observation, self.off_phase)

    def make_counts(self, dataset, observation):
        """Make off counts.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Input dataset.
        observation : `DatastoreObservation`
            Data store observation.

        Returns
        -------
        counts_off : `CountsSpectrum`
            Off counts.
        """
        return self._make_counts(dataset, observation, self.on_phase)

    def run(self, dataset, observation):
        """Run all steps.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Input dataset.
        observation : `DataStoreObservation`
            Data store observation.

        Returns
        -------
        dataset_on_off : `SpectrumDatasetOnOff`
            On off dataset.
        """
        from gammapy.datasets.dataset import SpectrumDatasetOnOff

        counts_off = self.make_counts_off(dataset, observation)
        counts = self.make_counts(dataset, observation)
        acceptance = np.sum([_[1] - _[0] for _ in self.on_phase])
        acceptance_off = np.sum([_[1] - _[0] for _ in self.off_phase])

        return SpectrumDatasetOnOff(
            counts=counts,
            counts_off=counts_off,
            gti=dataset.gti,
            name=dataset.name,
            livetime=dataset.livetime,
            edisp=dataset.edisp,
            aeff=dataset.aeff,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
        )

    @staticmethod
    def _check_intervals(intervals):
        """Split phase intervals that go beyond phase 1"""
        if isinstance(intervals, tuple):
            intervals = [intervals]

        for phase_interval in intervals:
            if phase_interval[0] > phase_interval[1]:
                intervals.remove(phase_interval)
                intervals.append([phase_interval[0], 1])
                intervals.append([0, phase_interval[1]])
        return intervals

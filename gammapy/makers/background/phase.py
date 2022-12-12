# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.data import EventList
from gammapy.datasets import (
    MapDataset,
    MapDatasetOnOff,
    SpectrumDataset,
    SpectrumDatasetOnOff,
)
from gammapy.maps import Map
from ..core import Maker

__all__ = ["PhaseBackgroundMaker"]


class PhaseBackgroundMaker(Maker):
    """Background estimation with on and off phases.

    TODO: For a usage example see future notebook.

    Parameters
    ----------
    on_phase : `tuple` or list of tuples
        on-phase defined by the two edges of each interval (edges are excluded)
    off_phase : `tuple` or list of tuples
        off-phase defined by the two edges of each interval (edges are excluded)
    column_name : `str`
        The name of the column in the event file from which the phase informations are extracted
    """

    tag = "PhaseBackgroundMaker"

    def __init__(self, on_phase, off_phase, column_name="PHASE"):
        self.on_phase = self._check_intervals(on_phase)
        self.off_phase = self._check_intervals(off_phase)
        self.column_name = column_name

    def __str__(self):
        s = self.__class__.__name__
        s += f"\n{self.on_phase}"
        s += f"\n{self.off_phase}"
        s += f"\n{self.column_name}"
        return s

    @staticmethod
    def _make_counts(dataset, observation, phases, column_name):

        event_lists = []
        for interval in phases:
            events = observation.events.select_parameter(
                parameter=column_name, band=interval
            )
            event_lists.append(events)

        events = EventList.from_stack(event_lists)
        counts = Map.from_geom(dataset.counts.geom)
        counts.fill_events(events)
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
        counts_off : `RegionNDMap`
            Off counts.
        """
        return self._make_counts(dataset, observation, self.off_phase, self.column_name)

    def make_counts(self, dataset, observation):
        """Make on counts.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Input dataset.
        observation : `DatastoreObservation`
            Data store observation.

        Returns
        -------
        counts : `RegionNDMap`
            On counts.
        """
        return self._make_counts(dataset, observation, self.on_phase, self.column_name)

    def run(self, dataset, observation):
        """Run all steps.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Input dataset.
        observation : `Observation`
            Data store observation.

        Returns
        -------
        dataset_on_off : `SpectrumDatasetOnOff`
            On off dataset.
        """
        counts_off = self.make_counts_off(dataset, observation)
        counts = self.make_counts(dataset, observation)

        acceptance = Map.from_geom(geom=dataset.counts.geom)
        acceptance.data = np.sum([_[1] - _[0] for _ in self.on_phase])

        acceptance_off = Map.from_geom(geom=dataset.counts.geom)
        acceptance_off.data = np.sum([_[1] - _[0] for _ in self.off_phase])

        if isinstance(dataset, SpectrumDataset):
            dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
                dataset=dataset,
                counts_off=counts_off,
                acceptance=acceptance,
                acceptance_off=acceptance_off,
            )
            dataset_on_off.counts = counts
            return dataset_on_off

        elif isinstance(dataset, MapDataset):
            dataset_on_off = MapDatasetOnOff.from_map_dataset(
                dataset=dataset,
                counts_off=counts_off,
                acceptance=acceptance,
                acceptance_off=acceptance_off,
            )
            dataset_on_off.counts = counts
            return dataset_on_off

    @staticmethod
    def _check_intervals(intervals):
        """Split phase intervals that go below phase 0 and above phase 1"""
        if isinstance(intervals, tuple):
            intervals = [intervals]

        for phase_interval in intervals:
            if phase_interval[0] % 1 > phase_interval[1] % 1:
                intervals.remove(phase_interval)
                intervals.append([phase_interval[0] % 1, 1])
                intervals.append([0, phase_interval[1] % 1])
        return intervals

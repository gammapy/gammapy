# Licensed under a 3-clause BSD style license - see LICENSE.rst
from copy import deepcopy
import numpy as np
from regions import PointSkyRegion
from gammapy.data import EventList
from gammapy.datasets import MapDatasetOnOff, SpectrumDataset
from gammapy.makers.utils import make_counts_rad_max
from gammapy.maps import Map
from ..core import Maker

__all__ = ["PhaseBackgroundMaker"]


class PhaseBackgroundMaker(Maker):
    """Background estimation with on and off phases.

    Parameters
    ----------
    on_phase : `tuple` or list of tuples
        On-phase defined by the two edges of each interval (edges are excluded).
    off_phase : `tuple` or list of tuples
        Off-phase defined by the two edges of each interval (edges are excluded).
    phase_column_name : `str`, optional
        The name of the column in the event file from which the phase information are extracted.
        Default is 'PHASE'.
    """

    tag = "PhaseBackgroundMaker"

    def __init__(self, on_phase, off_phase, phase_column_name="PHASE"):
        self.on_phase = self._check_intervals(deepcopy(on_phase))
        self.off_phase = self._check_intervals(deepcopy(off_phase))
        self.phase_column_name = phase_column_name

    def __str__(self):
        s = self.__class__.__name__
        s += f"\nOn phase interval : {self.on_phase}"
        s += f"\nOff phase interval : {self.off_phase}"
        s += f"\nPhase column name : {self.phase_column_name}"
        return s

    @staticmethod
    def _make_counts(dataset, observation, phases, phase_column_name):

        event_lists = []
        for interval in phases:
            events = observation.events.select_parameter(
                parameter=phase_column_name, band=interval
            )
            event_lists.append(events)

        events = EventList.from_stack(event_lists)
        geom = dataset.counts.geom
        if geom.is_region and isinstance(geom.region, PointSkyRegion):
            counts = make_counts_rad_max(geom, observation.rad_max, events)
        else:
            counts = Map.from_geom(geom)
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
        return self._make_counts(
            dataset, observation, self.off_phase, self.phase_column_name
        )

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
        return self._make_counts(
            dataset, observation, self.on_phase, self.phase_column_name
        )

    def run(self, dataset, observation):
        """Make on off dataset.

        Parameters
        ----------
        dataset : `SpectrumDataset` or `MapDataset`
            Input dataset.
        observation : `Observation`
            Data store observation.

        Returns
        -------
        dataset_on_off : `SpectrumDatasetOnOff` or `MapDatasetOnOff`
            On off dataset.
        """
        counts_off = self.make_counts_off(dataset, observation)
        counts = self.make_counts(dataset, observation)

        acceptance = Map.from_geom(geom=dataset.counts.geom)
        acceptance.data = np.sum([_[1] - _[0] for _ in self.on_phase])

        acceptance_off = Map.from_geom(geom=dataset.counts.geom)
        acceptance_off.data = np.sum([_[1] - _[0] for _ in self.off_phase])

        dataset_on_off = MapDatasetOnOff.from_map_dataset(
            dataset=dataset,
            counts_off=counts_off,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
        )
        dataset_on_off.counts = counts

        if isinstance(dataset, SpectrumDataset):
            dataset_on_off = dataset_on_off.to_spectrum_dataset(dataset._geom.region)

        return dataset_on_off

    @staticmethod
    def _check_intervals(intervals):
        """Split phase intervals that go below phase 0 and above phase 1.

        Parameters
        ----------
        intervals: `tuple`or list of `tuple`
            Phase interval or list of phase intervals to check.

        Returns
        -------
        intervals: list of `tuple`
            Phase interval checked.
        """
        if isinstance(intervals, tuple):
            intervals = [intervals]

        for phase_interval in intervals:
            if phase_interval[0] % 1 > phase_interval[1] % 1:
                intervals.remove(phase_interval)
                intervals.append((phase_interval[0] % 1, 1))
                if phase_interval[1] % 1 != 0:
                    intervals.append((0, phase_interval[1] % 1))
        return intervals

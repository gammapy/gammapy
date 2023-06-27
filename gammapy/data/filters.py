# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
from astropy.table import unique, vstack
import pandas as pd

__all__ = ["ObservationFilter"]

log = logging.getLogger(__name__)


class ObservationFilter:
    """Holds and applies filters to observation data.

    Parameters
    ----------
    time_filter : `astropy.time.Time`
        Start and stop time of the selected time interval. For now we only support
        a single time interval.
    event_filters : list of dict
        An event filter dictionary needs two keys:

        - **type** : str, one of the keys in `~gammapy.data.ObservationFilter.EVENT_FILTER_TYPES`
        - **opts** : dict, it is passed on to the method of the `~gammapy.data.EventListBase`
          class that corresponds to the filter type
          (see `~gammapy.data.ObservationFilter.EVENT_FILTER_TYPES`)

    event_filter_method: str
        The filterring method to use on the event_filters. Either "intersect" or "union". Default is "intersect".
        If "intersect", the filtered event list will be the intersection of all the event filters.
        If "union", the filtered event list will be the union of all the event fileters, under uniquness condition.

    Examples
    --------
    >>> from gammapy.data import ObservationFilter, DataStore, Observation
    >>> from astropy.time import Time
    >>> from astropy.coordinates import Angle
    >>>
    >>> time_filter = Time(['2021-03-27T20:10:00', '2021-03-27T20:20:00'])
    >>> phase_filter = {'type': 'custom', 'opts': dict(parameter='PHASE', band=(0.2, 0.8))}
    >>>
    >>> my_obs_filter = ObservationFilter(time_filter=time_filter, event_filters=[phase_filter])
    >>>
    >>> ds = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")
    >>> my_obs = ds.obs(obs_id=111630)
    >>> my_obs.obs_filter = my_obs_filter
    """

    EVENT_FILTER_TYPES = dict(sky_region="select_region", custom="select_parameter")

    def __init__(
        self, time_filter=None, event_filters=None, event_filter_method="intersect"
    ):
        self.time_filter = time_filter
        self.event_filters = self._check_overlap_phase(event_filters) or []
        self.event_filter_method = event_filter_method

    @property
    def livetime_fraction(self):
        """Fraction of the livetime kept when applying the event_filters."""
        return self._check_filter_phase(self.event_filters)

    def filter_events(self, events):
        """Apply filters to an event list.

        Parameters
        ----------
        events : `~gammapy.data.EventListBase`
            Event list to which the filters will be applied

        Returns
        -------
        filtered_events : `~gammapy.data.EventListBase`
            The filtered event list
        """
        from gammapy.data import EventList

        filtered_events = self._filter_by_time(events)

        if self.event_filter_method == "intersect":

            for f in self.event_filters:
                method_str = self.EVENT_FILTER_TYPES[f["type"]]
                filtered_events = getattr(filtered_events, method_str)(**f["opts"])

            return filtered_events

        elif self.event_filter_method == "union":

            filtered_events = []
            for f in self.event_filters:
                method_str = self.EVENT_FILTER_TYPES[f["type"]]
                filtered_events.append(getattr(events, method_str)(**f["opts"]).table)

            table = unique(
                vstack(filtered_events, join_type="exact").sort("TIME"), keys="TIME"
            )
            tot_filtered_events = EventList(table)
            return tot_filtered_events

        else:
            raise ValueError(
                "event_filter_method has to be either 'intersect' or 'union'."
            )

    def filter_gti(self, gti):
        """Apply filters to a GTI table.

        Parameters
        ----------
        gti : `~gammapy.data.GTI`
            GTI table to which the filters will be applied

        Returns
        -------
        filtered_gti : `~gammapy.data.GTI`
            The filtered GTI table
        """
        return self._filter_by_time(gti)

    def _filter_by_time(self, data):
        """Returns a new time filtered data object.

        Calls the `select_time` method of the data object.
        """
        if self.time_filter:
            return data.select_time(self.time_filter)
        else:
            return data

    def copy(self):
        """Copy the `ObservationFilter` object."""
        return copy.deepcopy(self)

    @staticmethod
    def _check_filter_phase(event_filter):
        fraction = 0
        for f in event_filter:
            if f.get("opts").get("parameter") == "PHASE":
                band = f.get("opts").get("band")
                fraction += band[1] - band[0]

        return 1 if fraction == 0 else fraction

    @staticmethod
    def _check_overlap_phase(event_filter):
        bands = []
        for f in event_filter:
            if f.get("opts").get("parameter") == "PHASE":
                bands.append(f.get("opts").get("band"))

        intervals = pd.arrays.IntervalArray.from_tuples(bands)
        interval_matrix = []
        for b in bands:
            interval_matrix.append(intervals.overlaps(pd.Interval(b[0], b[1])))

        interval_matrix = np.array(interval_matrix)
        overlap_array = interval_matrix[~np.eye(interval_matrix.shape[0], dtype=bool)]
        if True in overlap_array:
            raise ValueError(
                "Overlapping bands in event_filters that apply to pulsar phase are not allowed."
            )

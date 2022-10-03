# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging

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

        The filtered event list will be an intersection of all filters. A union
        of filters is not supported yet.

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

    def __init__(self, time_filter=None, event_filters=None):
        self.time_filter = time_filter
        self.event_filters = event_filters or []

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
        filtered_events = self._filter_by_time(events)

        for f in self.event_filters:
            method_str = self.EVENT_FILTER_TYPES[f["type"]]
            filtered_events = getattr(filtered_events, method_str)(**f["opts"])

        return filtered_events

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

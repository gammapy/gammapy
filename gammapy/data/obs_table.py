# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import namedtuple
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.units import Quantity, Unit
from astropy.utils import lazyproperty
from gammapy.utils.regions import SphericalCircleSkyRegion
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import Checker
from gammapy.utils.time import time_ref_from_dict

__all__ = ["ObservationTable"]


class ObservationTable(Table):
    """Observation table.

    Data format specification: :ref:`gadf:obs-index`
    """

    @classmethod
    def read(cls, filename, **kwargs):
        """Read an observation table from file.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        """
        return super().read(make_path(filename), **kwargs)

    @property
    def pointing_radec(self):
        """Pointing positions as ICRS (`~astropy.coordinates.SkyCoord`)."""
        return SkyCoord(self["RA_PNT"], self["DEC_PNT"], unit="deg", frame="icrs")

    @property
    def pointing_galactic(self):
        """Pointing positions as Galactic (`~astropy.coordinates.SkyCoord`)."""
        return SkyCoord(
            self["GLON_PNT"], self["GLAT_PNT"], unit="deg", frame="galactic"
        )

    @property
    def time_ref(self):
        """Time reference (`~astropy.time.Time`)."""
        return time_ref_from_dict(self.meta)

    @property
    def time_start(self):
        """Observation start time (`~astropy.time.Time`)."""
        return self.time_ref + Quantity(self["TSTART"], "second")

    @property
    def time_stop(self):
        """Observation stop time (`~astropy.time.Time`)."""
        return self.time_ref + Quantity(self["TSTOP"], "second")

    @lazyproperty
    def _index_dict(self):
        """Dict containing row index for all obs ids."""
        # TODO: Switch to http://docs.astropy.org/en/latest/table/indexing.html once it is more stable
        temp = zip(self["OBS_ID"], np.arange(len(self)))
        return dict(temp)

    def get_obs_idx(self, obs_id):
        """Get row index for given ``obs_id``.

        Raises KeyError if observation is not available.

        Parameters
        ----------
        obs_id : int, list
            observation ids

        Returns
        -------
        idx : list
            indices corresponding to obs_id
        """
        idx = [self._index_dict[key] for key in np.atleast_1d(obs_id)]
        return idx

    def select_obs_id(self, obs_id):
        """Get `~gammapy.data.ObservationTable` containing only ``obs_id``.

        Raises KeyError if observation is not available.

        Parameters
        ----------
        obs_id: int, list
            observation ids
        """
        return self[self.get_obs_idx(obs_id)]

    def summary(self):
        """Summary info string (str)."""
        obs_name = self.meta.get("OBSERVATORY_NAME", "N/A")
        return (
            f"Observation table:\n"
            f"Observatory name: {obs_name!r}\n"
            f"Number of observations: {len(self)}\n"
        )

    def select_range(self, selection_variable, value_range, inverted=False):
        """Make an observation table, applying some selection.

        Generic function to apply a 1D box selection (min, max) to a
        table on any variable that is in the observation table and can
        be casted into a `~astropy.units.Quantity` object.

        If the range length is 0 (min = max), the selection is applied
        to the exact value indicated by the min value. This is useful
        for selection of exact values, for instance in discrete
        variables like the number of telescopes.

        If the inverted flag is activated, the selection is applied to
        keep all elements outside the selected range.

        Parameters
        ----------
        selection_variable : str
            Name of variable to apply a cut (it should exist on the table).
        value_range : `~astropy.units.Quantity`-like
            Allowed range of values (min, max). The type should be
            consistent with the selection_variable.
        inverted : bool, optional
            Invert selection: keep all entries outside the (min, max) range.

        Returns
        -------
        obs_table : `~gammapy.data.ObservationTable`
            Observation table after selection.
        """
        value_range = Quantity(value_range)

        # read values into a quantity in case units have to be taken into account
        value = Quantity(self[selection_variable])

        mask = (value_range[0] <= value) & (value < value_range[1])

        if np.allclose(value_range[0].value, value_range[1].value):
            mask = value_range[0] == value

        if inverted:
            mask = np.invert(mask)

        return self[mask]

    def select_time_range(self, time_range, partial_overlap=False, inverted=False):
        """Make an observation table, applying a time selection.

        Apply a 1D box selection (min, max) to a
        table on any time variable that is in the observation table.
        It supports absolute times in `~astropy.time.Time` format.

        If the inverted flag is activated, the selection is applied to
        keep all elements outside the selected range.

        Parameters
        ----------
        time_range : `~astropy.time.Time`
            Allowed time range (min, max).
        partial_overlap : bool, optional
            Include partially overlapping observations. Default is False
        inverted : bool, optional
            Invert selection: keep all entries outside the (min, max) range.

        Returns
        -------
        obs_table : `~gammapy.data.ObservationTable`
            Observation table after selection.
        """
        tstart = self.time_start
        tstop = self.time_stop

        if partial_overlap is False:
            mask1 = time_range[0] <= tstart
            mask2 = time_range[1] >= tstop
        else:
            mask1 = time_range[0] <= tstop
            mask2 = time_range[1] >= tstart

        mask = mask1 & mask2

        if inverted:
            mask = np.invert(mask)

        return self[mask]

    def select_observations(self, selection=None):
        """Select subset of observations.

        Returns a new observation table representing the subset.

        There are 3 main kinds of selection criteria, according to the
        value of the **type** keyword in the **selection** dictionary:

        - sky regions

        - time intervals (min, max)

        - intervals (min, max) on any other parameter present in the
          observation table, that can be casted into an
          `~astropy.units.Quantity` object

        Allowed selection criteria are interpreted using the following
        keywords in the **selection** dictionary under the **type** key.

        - ``sky_circle`` is a circular region centered in the coordinate
           marked by the **lon** and **lat** keywords, and radius **radius**;
           uses `~gammapy.catalog.select_sky_circle`

        - ``time_box`` is a 1D selection criterion acting on the observation
          start time (**TSTART**); the interval is set via the
          **time_range** keyword; uses
          `~gammapy.data.ObservationTable.select_time_range`

        - ``par_box`` is a 1D selection criterion acting on any
          parameter defined in the observation table that can be casted
          into an `~astropy.units.Quantity` object; the parameter name
          and interval can be specified using the keywords **variable** and
          **value_range** respectively; min = max selects exact
          values of the parameter; uses
          `~gammapy.data.ObservationTable.select_range`

        In all cases, the selection can be inverted by activating the
        **inverted** flag, in which case, the selection is applied to keep all
        elements outside the selected range.

        A few examples of selection criteria are given below.

        Parameters
        ----------
        selection : dict
            Dictionary with a few keywords for applying selection cuts.

        Returns
        -------
        obs_table : `~gammapy.data.ObservationTable`
            Observation table after selection.

        Examples
        --------
        >>> selection = dict(type='sky_circle', frame='galactic',
        ...                  lon=Angle(0, 'deg'),
        ...                  lat=Angle(0, 'deg'),
        ...                  radius=Angle(5, 'deg'),
        ...                  border=Angle(2, 'deg'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='time_box',
        ...                  time_range=Time(['2012-01-01T01:00:00', '2012-01-01T02:00:00']))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='ALT',
        ...                  value_range=Angle([60., 70.], 'deg'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='OBS_ID',
        ...                  value_range=[2, 5])
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='N_TELS',
        ...                  value_range=[4, 4])
        >>> selected_obs_table = obs_table.select_observations(selection)
        """
        if "inverted" not in selection:
            selection["inverted"] = False
        if "partial_overlap" not in selection:
            selection["partial_overlap"] = False

        if selection["type"] == "sky_circle":
            lon = Angle(selection["lon"], "deg")
            lat = Angle(selection["lat"], "deg")
            radius = Angle(selection["radius"])
            if "border" in selection:
                border = Angle(selection["border"])
            else:
                border = Angle(0, "deg")
            region = SphericalCircleSkyRegion(
                center=SkyCoord(lon, lat, frame=selection["frame"]),
                radius=radius + border,
            )
            mask = region.contains(self.pointing_radec)
            if selection["inverted"]:
                mask = np.invert(mask)
            return self[mask]
        elif selection["type"] == "time_box":
            return self.select_time_range(
                selection["time_range"],
                selection["partial_overlap"],
                selection["inverted"],
            )
        elif selection["type"] == "par_box":
            return self.select_range(
                selection["variable"], selection["value_range"], selection["inverted"]
            )
        else:
            raise ValueError(f"Invalid selection type: {selection['type']}")


class ObservationTableChecker(Checker):
    """Event list checker.

    Data format specification: ref:`gadf:iact-events`

    Parameters
    ----------
    event_list : `~gammapy.data.EventList`
        Event list
    """

    CHECKS = {
        "meta": "check_meta",
        "columns": "check_columns",
        # "times": "check_times",
        # "coordinates_galactic": "check_coordinates_galactic",
        # "coordinates_altaz": "check_coordinates_altaz",
    }

    # accuracy = {"angle": Angle("1 arcsec"), "time": Quantity(1, "microsecond")}

    # https://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html#mandatory-header-keywords
    meta_required = [
        "HDUCLASS",
        "HDUDOC",
        "HDUVERS",
        "HDUCLAS1",
        "HDUCLAS2",
        # https://gamma-astro-data-formats.readthedocs.io/en/latest/general/time.html#time-formats
        "MJDREFI",
        "MJDREFF",
        "TIMEUNIT",
        "TIMESYS",
        "TIMEREF",
        # https://gamma-astro-data-formats.readthedocs.io/en/latest/general/coordinates.html#coords-location
        "GEOLON",
        "GEOLAT",
        "ALTITUDE",
    ]

    _col = namedtuple("col", ["name", "unit"])
    columns_required = [
        _col(name="OBS_ID", unit=""),
        _col(name="RA_PNT", unit="deg"),
        _col(name="DEC_PNT", unit="deg"),
        _col(name="TSTART", unit="s"),
        _col(name="TSTOP", unit="s"),
    ]

    def __init__(self, obs_table):
        self.obs_table = obs_table

    def _record(self, level="info", msg=None):
        return {"level": level, "hdu": "obs-index", "msg": msg}

    def check_meta(self):
        m = self.obs_table.meta

        meta_missing = sorted(set(self.meta_required) - set(m))
        if meta_missing:
            yield self._record(
                level="error", msg=f"Missing meta keys: {meta_missing!r}"
            )

        if m.get("HDUCLAS1", "") != "INDEX":
            yield self._record(level="error", msg="HDUCLAS1 must be INDEX")
        if m.get("HDUCLAS2", "") != "OBS":
            yield self._record(level="error", msg="HDUCLAS2 must be OBS")

    def check_columns(self):
        t = self.obs_table

        if len(t) == 0:
            yield self._record(level="error", msg="Observation table has zero rows")

        for name, unit in self.columns_required:
            if name not in t.colnames:
                yield self._record(level="error", msg=f"Missing table column: {name!r}")
            else:
                if Unit(unit) != (t[name].unit or ""):
                    yield self._record(
                        level="error", msg=f"Invalid unit for column: {name!r}"
                    )

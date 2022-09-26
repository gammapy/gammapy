# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import namedtuple
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.units import Quantity, Unit
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

    def select_obs_id(self, obs_id):
        """Get `~gammapy.data.ObservationTable` containing only ``obs_id``.

        Raises KeyError if observation is not available.

        Parameters
        ----------
        obs_id: int, list
            observation ids
        """
        try:
            self.indices["OBS_ID"]
        except IndexError:
            self.add_index("OBS_ID")
        return self.__class__(self.loc["OBS_ID", obs_id])

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

        if not partial_overlap:
            mask1 = time_range[0] <= tstart
            mask2 = time_range[1] >= tstop
        else:
            mask1 = time_range[0] <= tstop
            mask2 = time_range[1] >= tstart

        mask = mask1 & mask2

        if inverted:
            mask = np.invert(mask)

        return self[mask]

    def select_sky_circle(self, center, radius, inverted=False):
        """Make an observation table, applying a cone selection.

        Apply a selection based on the separation between the cone center
        and the observation pointing stored in the table.

        If the inverted flag is activated, the selection is applied to
        keep all elements outside the selected range.

        Parameters
        ----------
        center : `~astropy.coordinate.SkyCoord`
            Cone center coordinate.
        radius : `~astropy.coordinate.Angle`
            Cone opening angle. The maximal separation allowed between the center
            and the observation pointing direction.
        inverted : bool, optional
            Invert selection: keep all entries outside the cone.

        Returns
        -------
        obs_table : `~gammapy.data.ObservationTable`
            Observation table after selection.
        """
        region = SphericalCircleSkyRegion(center=center, radius=radius)
        mask = region.contains(self.pointing_radec)
        if inverted:
            mask = np.invert(mask)
        return self[mask]

    def select_observations(self, selections=None):
        """Select subset of observations from a list of selection criteria.

        Returns a new observation table representing the subset.

        There are 3 main kinds of selection criteria, according to the
        value of the **type** keyword in the **selection** dictionary:

        - circular region

        - time intervals (min, max)

        - intervals (min, max) on any other parameter present in the
          observation table, that can be casted into an
          `~astropy.units.Quantity` object

        Allowed selection criteria are interpreted using the following
        keywords in the **selection** dictionary under the **type** key.

        - ``sky_circle`` is a circular region centered in the coordinate
           marked by the **lon** and **lat** keywords, and radius **radius**

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
        selection : list of dict
            List of selection cuts dictionaries.

        Returns
        -------
        obs_table : `~gammapy.data.ObservationTable`
            Observation table after selection.

        Examples
        --------
        >>> from gammapy.data import ObservationTable
        >>> obs_table = ObservationTable.read('$GAMMAPY_DATA/hess-dl3-dr1/obs-index.fits.gz')
        >>> from astropy.coordinates import Angle
        >>> selection = dict(type='sky_circle', frame='galactic',
        ...                  lon=Angle(0, 'deg'),
        ...                  lat=Angle(0, 'deg'),
        ...                  radius=Angle(5, 'deg'),
        ...                  border=Angle(2, 'deg'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> from astropy.time import Time
        >>> time_range = Time(['2012-01-01T01:00:00', '2012-01-01T02:00:00'])
        >>> selection = dict(type='time_box', time_range=time_range)
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> value_range = Angle([60., 70.], 'deg')
        >>> selection = dict(type='par_box', variable='ALT_PNT', value_range=value_range)
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='OBS_ID', value_range=[2, 5])
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='N_TELS', value_range=[4, 4])
        >>> selected_obs_table = obs_table.select_observations(selection)
        """
        if isinstance(selections, dict):
            selections = [selections]

        obs_table = self
        for selection in selections:
            obs_table = obs_table._apply_simple_selection(selection)

        return obs_table

    def _apply_simple_selection(self, selection):
        """Select subset of observations from a single selection criterion."""
        selection = selection.copy()
        type = selection.pop("type")
        if type == "sky_circle":
            lon = Angle(selection.pop("lon"), "deg")
            lat = Angle(selection.pop("lat"), "deg")
            radius = Angle(selection.pop("radius"), "deg")
            radius += Angle(selection.pop("border", 0), "deg")
            center = SkyCoord(lon, lat, frame=selection.pop("frame"))
            return self.select_sky_circle(center, radius, **selection)
        elif type == "time_box":
            time_range = selection.pop("time_range")
            return self.select_time_range(time_range, **selection)
        elif type == "par_box":
            variable = selection.pop("variable")
            return self.select_range(variable, **selection)
        else:
            raise ValueError(f"Invalid selection type: {type}")


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

    @staticmethod
    def _record(level="info", msg=None):
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

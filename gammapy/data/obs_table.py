# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import namedtuple
import numpy as np
from astropy.coordinates import Angle, SkyCoord, EarthLocation
from astropy.table import Table, Column
from astropy.units import Quantity, Unit
from gammapy.utils.regions import SphericalCircleSkyRegion
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import Checker
from gammapy.utils.time import time_ref_from_dict
from gammapy.data.metadata import METADATA_FITS_KEYS
from astropy.time import Time
from astropy import units as u

__all__ = ["ObservationTable"]


class ObservationTable(Table):
    """Modified ObservationTable class, based on existing ObservationTable class.

    See discussion and development: https://github.com/gammapy/gammapy/issues/3767, https://github.com/gammapy/gammapy/issues/4238
    Co-authors: @maxnoe, @registerrier, @bkhelifi
    Used as reference: gammapy, gammapy/data/obs_table.py, https://docs.python.org/3, https://docs.astropy.org/en/latest/table/construct_table.html#construct-table, https://numpy.org/doc/stable/reference/generated/numpy.dtype.html
                       https://docs.astropy.org/en/latest/table/index.html, https://gamma-astro-data-formats.readthedocs.io/en/v0.3/, esp. data_storage/obs_index/index.html, https://www.programiz.com/python-programming/methods/built-in/classmethod
    Looked into: https://github.com/gammasky/cta-dc/blob/master/data/cta_1dc_make_data_index_files.py, maybe used l. 233. Copyright (c) 2016 gammasky,
    Oriented also at PR by @registerrier: https://github.com/gammapy/gammapy/pull/5954/files

    # ATTRIBUTION copied from hess-dl3-dr1 README.txt (gammapy-data/hess-dl3-dr1/README.txt) for testing and learning from this dataset:
    # This work made use of data from the H.E.S.S. DL3 public test
    # data release 1 (HESS DL3 DR1, H.E.S.S. collaboration, 2018).
    """

    def __init__(self):
        """Constructor for internal observation table.

         Parameters
        ----------
        table : `astropy.table.Table'
            Table to init ObservationTable from.

        Creates instance of ObservationTable either from reference table.
        """
        # Used for constructor: https://stackoverflow.com/questions/6535832/python-inherit-the-superclass-init

        # Init with basic reference table, like suggested by @registerrier.
        super(ObservationTable, self).__init__(self._reference_table())

    @staticmethod
    def _reference_table():
        """Definition of internal observation table model in form of reference table object."""

        table = Table(
            [
                Column(
                    name="OBS_ID",
                    unit=None,
                    description="Observation ID per observation run",
                    dtype=str,
                ),
                Column(
                    name="OBJECT",
                    unit=None,
                    description="Name of the object",
                    dtype=str,
                ),
            ]
        )
        table["POINTING"] = SkyCoord([], [], unit=u.deg, frame="icrs")
        table["LOCATION"] = EarthLocation.from_geodetic([], [], [])
        table["TSTART"] = Time([], format="mjd")
        table["TSTOP"] = Time([], format="mjd")

        return table

    def read(self, filename, fileformat=None, **kwargs):
        """Modified reader for ObservationTable"""
        """Header and super().read(make_path(filename), **kwargs) modified from legacy class ObservationTable."""

        """Read an observation table from file.

        Parameters
        ----------
        filename : `pathlib.Path` or str
            Filename.
        fileformat : str
            Fileformat, default is "GADF0.3" for GADF v.0.3.
        **kwargs : dict, optional
            Keyword arguments passed to `~astropy.table.Table.read`.
        """

        # Read disk table "table_disk", taken from class ObervationTable. TODO: Pot. lazy loading in future?"""
        table_disk = Table.read(make_path(filename), **kwargs)

        # Get header of obs-index table.
        meta = table_disk.meta

        # If no file-format specified, try to infer file format of table_disk, otherwise use GADF v.0.3. As discussed with @bkhelifi.
        if fileformat is None:
            if "HDUCLASS" in meta.keys():
                if "HDUVERS" in meta.keys():
                    fileformat = meta["HDUCLASS"] + meta["HDUVERS"]
            else:
                fileformat = "GADF0.3"  # Use default "GADF0.3".

        # For specified fileformat call reader to convert to internal data model, as discussed with @bkhelifi, @registerrier.
        if fileformat == "GADF0.2":
            return self.read_from_gadf02(table_disk)
        elif fileformat == "GADF0.3":
            return self.read_from_gadf03(table_disk)

    def read_from_gadf03(self, table_disk):
        """Converter from GADF v.0.3 to internal table model."""
        """ Based on specification: https://gamma-astro-data-formats.readthedocs.io/en/v0.3/"""

        # Create internal table "table_internal" with all names, corresp. units, types and descriptions, for the internal table model.
        table_internal = self.table

        return table_internal

    def read_from_gadf02(self, table_disk):
        """Converter from GADF v.0.2 to internal table model."""
        """Based on specification: https://gamma-astro-data-formats.readthedocs.io/en/v0.2/"""

        # Required names to fill internal table format, for GADF v.0.2, will be extended after checking POINTING. Similar to PR#5954.
        required_names_on_disk = [
            "OBS_ID",
            "OBJECT",
            "GEOLON",
            "GEOLAT",
            "ALTITUDE",
            "TSTART",
            "TSTOP",
        ]

        # Get colnames of disk_table
        names_disk = table_disk.colnames

        # Get header of obs-index table.
        meta = table_disk.meta

        # Check which info is given for POINTING
        # Used commit 16ce9840f38bea55982d2cd986daa08a3088b434 by @registerrier
        if "OBS_MODE" in names_disk:
            # like in data_store.py:
            if table_disk["OBS_MODE"] == "DRIFT":
                required_names_on_disk.append("ALT_PNT")
                required_names_on_disk.append("AZ_PNT")
            else:
                required_names_on_disk.append("RA_PNT")
                required_names_on_disk.append("DEC_PNT")
        else:
            # if "OBS_MODE" not given, decide based on what is given, RADEC or ALTAZ
            if "RA_PNT" in names_disk:
                required_names_on_disk.append("RA_PNT")
                required_names_on_disk.append("DEC_PNT")
            elif "ALT_PNT" in names_disk:
                required_names_on_disk.append("ALT_PNT")
                required_names_on_disk.append("AZ_PNT")
            else:
                raise RuntimeError("Neither RADEC nor ALTAZ is given in table on disk!")

        # Used: aeb1ea01e60e1f02c5fb59f50141c81e0b2fb8f6:
        missing_names = set(required_names_on_disk).difference(
            names_disk + list(meta.keys())
        )
        if len(missing_names) != 0:
            raise RuntimeError(
                f"Not all columns required for GADF v.0.2 were found in file. Missing: {missing_names}"
            )
        #             )  # looked into gammapy/workflow/core.py

        # Create internal table "table_internal" with all names, corresp. units, types and descriptions, for the internal table model.
        table_internal = self

        # Fill internal table for mandatory columns by constructing the table row-wise with the internal representations.
        number_of_observations = len(
            table_disk
        )  # Get number of observations, equal to number of rows in table on disk.

        for i in range(number_of_observations):
            row_internal = [str(table_disk[i]["OBS_ID"]), str(table_disk[i]["OBJECT"])]

            if "RA_PNT" in required_names_on_disk:
                row_internal.append(
                    METADATA_FITS_KEYS["pointing"]["radec_mean"]["input"](
                        {
                            "RA_PNT": table_disk[i]["RA_PNT"],
                            "DEC_PNT": table_disk[i]["DEC_PNT"],
                        }
                    )
                )
            elif "ALT_PNT" in required_names_on_disk:
                row_internal.append(
                    METADATA_FITS_KEYS["pointing"]["altaz_mean"]["input"](
                        {
                            "ALT_PNT": table_disk[i]["ALT_PNT"],
                            "AZ_PNT": table_disk[i]["AZ_PNT"],
                        }
                    )
                )

            row_internal.append(
                METADATA_FITS_KEYS["observation"]["location"]["input"](
                    {
                        "GEOLON": meta["GEOLON"],
                        "GEOLAT": meta["GEOLAT"],
                        "ALTITUDE": meta["ALTITUDE"],
                    }
                )
            )

            # from @properties "time_ref", "time_start", "time_stop"
            time_ref = time_ref_from_dict(meta)
            row_internal.append(time_ref + Quantity(table_disk[i]["TSTART"], "second"))
            row_internal.append(time_ref + Quantity(table_disk[i]["TSTOP"], "second"))

            # )  # like in event_list.py, l.201, commit: 08c6f6a
            table_internal.add_row(
                row_internal
            )  # Add row to internal table (fill table).

        # Load optional columns, whose names are not already processed, automatically into internal table.
        opt_names = set(names_disk).difference(required_names_on_disk)
        for name in opt_names:  # add column-wise all optional column-data present in file, independent of format.
            table_internal[name] = table_disk[name]

        # return internal table, instead of copy of disk-table like before.
        return table_internal

    @property
    def pointing_radec(self):
        """Pointing positions in ICRS as a `~astropy.coordinates.SkyCoord` object."""
        return SkyCoord(self["RA_PNT"], self["DEC_PNT"], unit="deg", frame="icrs")

    @property
    def pointing_galactic(self):
        """Pointing positions in Galactic coordinates as a `~astropy.coordinates.SkyCoord` object."""
        return SkyCoord(
            self["GLON_PNT"], self["GLAT_PNT"], unit="deg", frame="galactic"
        )

    @property
    def time_ref(self):
        """Time reference as a `~astropy.time.Time` object."""
        return time_ref_from_dict(self.meta)

    @property
    def time_start(self):
        """Observation start time as a `~astropy.time.Time` object."""
        return self.time_ref + Quantity(self["TSTART"], "second")

    @property
    def time_stop(self):
        """Observation stop time as a `~astropy.time.Time` object."""
        return self.time_ref + Quantity(self["TSTOP"], "second")

    def select_obs_id(self, obs_id):
        """Get `~gammapy.data.ObservationTable` containing only ``obs_id``.

        Raises KeyError if observation is not available.

        Parameters
        ----------
        obs_id : int or list of int
            Observation ids.
        """
        try:
            self.indices["OBS_ID"]
        except IndexError:
            self.add_index("OBS_ID")
        return self.__class__(self.loc["OBS_ID", obs_id])

    def summary(self):
        """Summary information string."""
        obs_name = self.meta.get(
            "OBSERVATORY_NAME", "N/A"
        )  # This is not GADF compliant
        if "N/A" in obs_name:
            obs_name = self.meta.get(
                "OBSERVATORY_NAME", self.meta.get("OBSERVER", "N/A")
            )

        return (
            f"Observation table:\n"
            f"Observatory name: {obs_name!r}\n"
            f"Number of observations: {len(self)}\n"
        )

    def select_range(self, selection_variable, value_range, inverted=False):
        """Make an observation table, applying some selection.

        Generic function to apply a 1D box selection (min, max) to a
        table on any variable that is in the observation table and can
        be cast into a `~astropy.units.Quantity` object.

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
            Default is False.

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
            Include partially overlapping observations. Default is False.
        inverted : bool, optional
            Invert selection: keep all entries outside the (min, max) range.
            Default is False.

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
        center : `~astropy.coordinates.SkyCoord`
            Cone center coordinate.
        radius : `~astropy.coordinates.Angle`
            Cone opening angle. The maximal separation allowed between the center
            and the observation pointing direction.
        inverted : bool, optional
            Invert selection: keep all entries outside the cone. Default is False.

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
          observation table, that can be cast into an
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
        selections : list of dict, optional
            Dictionary of selection criteria. Default is None.

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

    Data format specification: ref:`gadf:iact-events`.

    Parameters
    ----------
    obs_table : `~gammapy.data.ObservationTable`
        Observation table.
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

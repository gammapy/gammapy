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
from gammapy.utils.scripts import read_yaml
from gammapy.utils.types import cast_func
from pathlib import Path
# from astropy.time import Time

__all__ = ["ObservationTable"]


class ObservationTable(Table):
    """Modified ObservationTable class, based on existing ObservationTable class.

    See discussion and development: https://github.com/gammapy/gammapy/issues/3767, https://github.com/gammapy/gammapy/issues/4238
    Co-authors: @maxnoe, @registerrier, @bkhelifi
    Used as reference: gammapy, gammapy/data/obs_table.py, https://docs.python.org/3, https://docs.astropy.org/en/latest/table/construct_table.html#construct-table, https://numpy.org/doc/stable/reference/generated/numpy.dtype.html
                       https://docs.astropy.org/en/latest/table/index.html, https://gamma-astro-data-formats.readthedocs.io/en/v0.3/, esp. data_storage/obs_index/index.html
    Looked into: https://github.com/gammasky/cta-dc/blob/master/data/cta_1dc_make_data_index_files.py, maybe used l. 233. Copyright (c) 2016 gammasky

    # ATTRIBUTION copied from hess-dl3-dr1 README.txt (gammapy-data/hess-dl3-dr1/README.txt) for testing and learning from this dataset:
    # This work made use of data from the H.E.S.S. DL3 public test
    # data release 1 (HESS DL3 DR1, H.E.S.S. collaboration, 2018).
    """

    # Required minimum names of internal table. These will be translated into needed names on disk, depending on the fileformat, in the reader.
    names_min_req = ["OBS_ID", "OBJECT", "POINTING"]

    @classmethod
    def read(cls, filename, fileformat=None, **kwargs):
        """Modified reader for ObservationTable"""
        """Header and super().read(make_path(filename), **kwargs) taken from legacy class ObservationTable."""

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
        table_disk = super().read(make_path(filename), **kwargs)

        # Get header of obs-index table.
        meta = table_disk.meta

        # If no file-format specified, try to infer file format of table_disk, otherwise use GADF v.0.3. As discussed with @bkhelifi.
        if fileformat is None:
            if "HDUCLASS" in meta.keys():
                if "HDUVERS" in meta.keys():
                    fileformat = meta["HDUCLASS"] + meta["HDUVERS"]
            else:
                fileformat = "GADF0.3"  # Use default "GADF0.3".

        # For the chosen fileformat, read then info on internal format and on correspondance to the selected fileformat.
        format = cls.get_format_dict(fileformat)
        names_internal = list(format.keys())

        # Get correspondance of internal names to (multiple) disk-names, called "correspondance_dict".
        # Also, flattened version in "correspondance_dict_flat", which will be used to check later, which disk-names are (truly) optional (and not already processed).
        correspondance_dict = {}
        correspondance_dict_flat = []

        for name in names_internal:
            correspondance_dict[name] = cls.get_corresponding_names(name, format)
            correspondance_dict_flat.extend(cls.get_corresponding_names(name, format))

        # Get corresponding names now only for minimal set of required names, to check if present on disk.
        # TODO: Adapt this for case of alternative names, e.g. for pointing.
        for name in cls.names_min_req:
            names_min_req_on_disk = correspondance_dict[name]
            for el in names_min_req_on_disk:
                if el not in table_disk.columns:
                    raise RuntimeError(
                        "Not all required names in "
                        + fileformat
                        + "-file found, first missed name: "
                        + el
                        + "."
                    )  # looked into gammapy/workflow/core.py

        # Create internal table "table_internal" with all names, corresp. units, types and descriptions, for the internal table model.
        # The internal table model may know more names than minimal required on disk for the read/fill-process.
        units_internal = []
        types_internal = []
        description_internal = []

        for name in names_internal:
            units_internal.append(format[name]["unit"])
            types_internal.append(format[name]["type"])
            description_internal.append(format[name]["description"])

        table_internal = cls(
            names=names_internal,
            units=units_internal,
            dtype=types_internal,
            descriptions=description_internal,
        )

        # Fill internal table for mandatory columns by constructing the table row-wise with the internal representations.
        number_of_observations = len(
            table_disk
        )  # Get number of observations, equal to number of rows in table on disk.
        for i in range(number_of_observations):
            row_internal = []
            for name in names_internal:
                names_disk = correspondance_dict[name]

                # Construction of in-mem representation of metadata.
                # Typecasting as noted by @bkhelifi for now here, by using function cast_func(value, type) in utils/types.py
                if name == "OBS_ID":
                    row_internal.append(
                        cast_func(
                            table_disk[i][names_disk[0]], np.dtype(format[name]["type"])
                        )
                    )
                elif name == "OBJECT":
                    row_internal.append(
                        cast_func(
                            table_disk[i][names_disk[0]], np.dtype(format[name]["type"])
                        )
                    )
                elif (
                    name == "POINTING"
                ):  # build object like @registerrier in 16ce9840f38bea55982d2cd986daa08a3088b434
                    row_internal.append(
                        SkyCoord(
                            cast_func(table_disk[i][names_disk[0]], np.dtype(float)),
                            cast_func(table_disk[i][names_disk[1]], np.dtype(float)),
                            unit="deg",
                            frame="icrs",
                        )
                    )
                # elif name == "TSTART":
                # row_internal.append(

                # time_ref_from_dict(table_disk.meta) + Time(table_disk[i][names_disk[0]],format="mjd",scale="tt"),
                # time_ref_from_dict(table_disk.meta) + Time(table_disk[i][names_disk[1]],format="mjd",scale="tt")

                # )
                # print(
                # time_ref_from_dict(meta)
                # )  # like in event_list.py, l.201, commit: 08c6f6a
            table_internal.add_row(
                row_internal
            )  # Add row to internal table (fill table).

        # Load optional columns, whose names are not already processed, automatically into internal table.
        opt_names = list(table_disk.columns)
        for name in correspondance_dict_flat:
            opt_names.remove(name)
        for name in opt_names:  # add column-wise all optional column-data present in file, independent of format.
            table_internal[name] = table_disk[name]

        # return internal table, instead of copy of disk-table like before.
        return table_internal

    def get_format_dict(fileformat):
        """Read info on the internal table format and its correspondance to the selected fileformat from a YAML-file.

        Parameters
        ----------
        fileformat : str
            Fileformat, default is "gadf03" for GADF v.0.3.

        Returns
        -------
        The loaded dictionary is returned as format.
        """
        PATH_FORMATS = (
            Path(__file__).resolve().parent / ".." / "utils" / "formats"
        )  # like gammapy/utils/scripts.py l.29, commit: 753fb3e
        format = read_yaml(str(PATH_FORMATS) + "/obs_index_" + fileformat + ".yaml")
        return format

    def get_corresponding_names(name, format):
        """For a given format and internal table name, get the corresponding disk-name(s).

        Parameters
        ----------
        name : str
            Column name of internal table-format.
        format : dict
            Dictionary containing the internal table-format definition and its correspondance to a fileformat.

        Returns
        -------
        List with the corresponding names per internal name.
        """

        n_disk_names = len(
            format[name]["disk"]
        )  # Get number of corresponding names on disk
        correspondance = []
        for n in range(n_disk_names):
            name_disk = format[
                name
            ][
                "disk"
            ][
                n
            ][
                "name"
            ]  # Get for the column(s) to be loaded the name(s) on disk, for selected fileformat.
            correspondance.append(name_disk)
        return correspondance

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

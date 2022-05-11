# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalog and object base classes."""
import abc
import numbers
from copy import deepcopy
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils import lazyproperty
from gammapy.maps import TimeMapAxis
from gammapy.modeling.models import Models
from gammapy.utils.table import table_from_row_data, table_row_to_dict

__all__ = ["SourceCatalog", "SourceCatalogObject"]


# https://pydanny.blogspot.com/2011/11/loving-bunch-class.html
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)


def format_flux_points_table(table):
    for column in table.colnames:
        if column.startswith(("dnde", "eflux", "flux", "e2dnde", "ref")):
            table[column].format = ".3e"
        elif column.startswith(
            ("e_min", "e_max", "e_ref", "sqrt_ts", "norm", "ts", "stat")
        ):
            table[column].format = ".3f"

    return table


class SourceCatalogObject:
    """Source catalog object.

    This class can be used directly, but it is mostly used as a
    base class for the other source catalog classes.

    The catalog data on this source is stored in the `source.data`
    attribute as a dict.

    The source catalog object is decoupled from the source catalog,
    it doesn't hold a reference back to it, except for a key
    ``_row_index`` of type ``int`` that links to the catalog table
    row the source information comes from.
    """

    _source_name_key = "Source_Name"
    _row_index_key = "_row_index"

    def __init__(self, data, data_extended=None):
        self.data = Bunch(**data)
        if data_extended:
            self.data_extended = Bunch(**data_extended)

    @property
    def name(self):
        """Source name (str)"""
        name = self.data[self._source_name_key]
        return name.strip()

    @property
    def row_index(self):
        """Row index of source in catalog (int)"""
        return self.data[self._row_index_key]

    @property
    def position(self):
        """Source position (`~astropy.coordinates.SkyCoord`)."""
        table = table_from_row_data([self.data])
        return _skycoord_from_table(table)[0]


class SourceCatalog(abc.ABC):
    """Generic source catalog.

    This class can be used directly, but it is mostly used as a
    base class for the other source catalog classes.

    This is a thin wrapper around `~astropy.table.Table`,
    which is stored in the ``catalog.table`` attribute.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with catalog data.
    source_name_key : str
        Column with source name information
    source_name_alias : tuple of str
        Columns with source name aliases. This will allow accessing the source
        row by alias names as well.
    """

    @classmethod
    @abc.abstractmethod
    def description(cls):
        """Catalog description (str)."""
        pass

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    source_object_class = SourceCatalogObject
    """Source class (`SourceCatalogObject`)."""

    def __init__(self, table, source_name_key="Source_Name", source_name_alias=()):
        self.table = table
        self._source_name_key = source_name_key
        self._source_name_alias = source_name_alias

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"    name: {self.tag}\n"
            f"    description: {self.description}\n"
            f"    sources: {len(self.table)}\n"
        )

    @lazyproperty
    def _name_to_index_cache(self):
        # Make a dict for quick lookup: source name -> row index
        names = {}
        for idx, row in enumerate(self.table):
            name = row[self._source_name_key]
            names[name.strip()] = idx
            for alias_column in self._source_name_alias:
                for alias in str(row[alias_column]).split(","):
                    if not alias == "":
                        names[alias.strip()] = idx
        return names

    def row_index(self, name):
        """Look up row index of source by name.

        Parameters
        ----------
        name : str
            Source name

        Returns
        -------
        index : int
            Row index of source in table
        """
        index = self._name_to_index_cache[name]
        row = self.table[index]
        # check if name lookup is correct other wise recompute _name_to_index_cache

        possible_names = [row[self._source_name_key]]
        for alias_column in self._source_name_alias:
            possible_names += str(row[alias_column]).split(",")

        if name not in possible_names:
            self.__dict__.pop("_name_to_index_cache")
            index = self._name_to_index_cache[name]

        return index

    def source_name(self, index):
        """Look up source name by row index.

        Parameters
        ----------
        index : int
            Row index of source in table
        """
        source_name_col = self.table[self._source_name_key]
        name = source_name_col[index]
        return name.strip()

    def __getitem__(self, key):
        """Get source by name.

        Parameters
        ----------
        key : str or int
            Source name or row index

        Returns
        -------
        source : `SourceCatalogObject`
            An object representing one source
        """
        if isinstance(key, str):
            index = self.row_index(key)
        elif isinstance(key, numbers.Integral):
            index = key
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            new = deepcopy(self)
            new.table = self.table[key]
            return new
        else:
            raise TypeError(f"Invalid key: {key!r}, {type(key)}\n")

        return self._make_source_object(index)

    def _make_source_object(self, index):
        """Make one source object.

        Parameters
        ----------
        index : int
            Row index

        Returns
        -------
        source : `SourceCatalogObject`
            Source object
        """
        data = table_row_to_dict(self.table[index])
        data[SourceCatalogObject._row_index_key] = index

        fp_energy_edges = getattr(self, "flux_points_energy_edges", None)

        if fp_energy_edges:
            data["fp_energy_edges"] = fp_energy_edges

        hist_table = getattr(self, "hist_table", None)
        hist2_table = getattr(self, "hist2_table", None)

        if hist_table:
            try:
                data["time_axis"] = TimeMapAxis.from_table(
                    hist_table, format="fermi-fgl"
                )
            except KeyError:
                pass

        if hist2_table:
            try:
                data["time_axis_2"] = TimeMapAxis.from_table(
                    hist2_table, format="fermi-fgl"
                )
            except KeyError:
                pass
        if "Extended_Source_Name" in data:
            name_extended = data["Extended_Source_Name"].strip()
        elif "Source_Name" in data:
            name_extended = data["Source_Name"].strip()
        else:
            name_extended = None
        try:
            idx = self._lookup_extended_source_idx[name_extended]
            data_extended = table_row_to_dict(self.extended_sources_table[idx])
        except (KeyError, AttributeError):
            data_extended = None

        source = self.source_object_class(data, data_extended)
        return source

    @lazyproperty
    def _lookup_extended_source_idx(self):
        names = [_.strip() for _ in self.extended_sources_table["Source_Name"]]
        idx = range(len(names))
        return dict(zip(names, idx))

    @property
    def positions(self):
        """Source positions (`~astropy.coordinates.SkyCoord`)."""
        return _skycoord_from_table(self.table)

    def to_models(self, **kwargs):
        """Create Models object from catalogue"""
        return Models([_.sky_model(**kwargs) for _ in self])


def _skycoord_from_table(table):
    keys = table.colnames

    if {"RAJ2000", "DEJ2000"}.issubset(keys):
        lon, lat, frame = "RAJ2000", "DEJ2000", "icrs"
    elif {"RA", "DEC"}.issubset(keys):
        lon, lat, frame = "RA", "DEC", "icrs"
    elif {"ra", "dec"}.issubset(keys):
        lon, lat, frame = "ra", "dec", "icrs"
    else:
        raise KeyError("No column GLON / GLAT or RA / DEC or RAJ2000 / DEJ2000 found.")

    unit = table[lon].unit.to_string() if table[lon].unit else "deg"

    return SkyCoord(table[lon], table[lat], unit=unit, frame=frame)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalog and object base classes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import sys
from pprint import pprint
from astropy.extern import six
from astropy.utils import lazyproperty
from astropy.units import Quantity
from ..utils.array import _is_int

__all__ = [
    'SourceCatalog',
    'SourceCatalogObject',
]


class SourceCatalogObject(object):
    """Source catalog object.

    This class can be used directly, but it's mostly used as a
    base class for the other source catalog classes.

    The catalog data on this source is stored in the `source.data`
    attribute as on OrderedDict.

    The source catalog object is decoupled from the source catalog,
    it doesn't hold a reference back to it.
    The catalog table row index is stored in `_table_row_index` though,
    because it can be useful for debugging or display.
    """
    _source_name_key = 'Source_Name'
    _source_index_key = 'catalog_row_index'

    def __init__(self, data):
        self.data = data

    @property
    def name(self):
        """Source name"""
        name = self.data[self._source_name_key]
        return name.strip()

    @property
    def index(self):
        """Row index of source in catalog"""
        return self.data[self._source_index_key]

    def pprint(self, file=None):
        """Pretty-print source data"""
        if not file:
            file = sys.stdout

        pprint(self.data, stream=file)

        # TODO: add methods to serialise to JSON and YAML
        # and also to quickly pretty-print output in that format for interactive use.
        # Maybe even add HTML output for IPython repr?
        # Or at to_table method?

    def info(self):
        """
        Print summary info about the object.
        """
        print(self)


class SourceCatalog(object):
    """Generic source catalog.

    This class can be used directly, but it's mostly used as a
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
    source_object_class = SourceCatalogObject

    # TODO: at the moment these are duplicated in SourceCatalogObject.
    # Should we share them somehow?
    _source_index_key = 'catalog_row_index'

    def __init__(self, table, source_name_key='Source_Name', source_name_alias=()):
        self.table = table
        self._source_name_key = source_name_key
        self._source_name_alias = source_name_alias

    @lazyproperty
    def _name_to_index_cache(self):
        # Make a dict for quick lookup: source name -> row index
        names = dict()
        for idx, row in enumerate(self.table):
            name = row[self._source_name_key]
            names[name.strip()] = idx
            for alias_column in self._source_name_alias:
                for alias in row[alias_column].split(','):
                    if not alias == '':
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
            possible_names += row[alias_column].split(',')

        if not name in possible_names:
            self.__dict__.pop('_name_to_index_cache')
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
            An object representing one source.

        Notes
        -----
        At the moment this can raise KeyError, IndexError and ValueError
        for invalid keys. Should we always raise KeyError to simplify this?
        """
        if isinstance(key, six.string_types):
            index = self.row_index(key)
        elif _is_int(key):
            index = key
        else:
            msg = 'Key must be source name string or row index integer. '
            msg += 'Type not understood: {}'.format(type(key))
            raise ValueError(msg)

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
        data = self._make_source_dict(index)
        source = self.source_object_class(data)
        return source

    def _make_source_dict(self, idx):
        """Make one source data dict.

        Parameters
        ----------
        idx : int
            Row index

        Returns
        -------
        data : `~collections.OrderedDict`
            Source data
        """
        data = OrderedDict()
        for colname in self.table.colnames:
            col = self.table[colname]

            if isinstance(col, Quantity):
                val = col[idx]
            else:
                val = col.data[idx]
                if col.unit:
                    val = Quantity(val, col.unit)

            data[colname] = val

        data[self._source_index_key] = idx
        return data

    def info(self):
        """Print info string."""
        print(self)

    def __str__(self):
        """Info string."""
        ss = self.description
        ss += ' with {} objects.'.format(len(self.table))
        return ss

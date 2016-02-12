# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalog and object base classes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import sys
from pprint import pprint
from astropy.extern import six

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
        self.data = OrderedDict(data)

    @property
    def name(self):
        """Source name"""
        name = self.data[self._source_name_key]
        return name.strip()

    @property
    def index(self):
        """Row index of source in catalog"""
        return self.data[self._source_index_key]

    def to_dict(self, cols=None):
        """
        Convert source data to Python dict.

        Parameters
        ----------
        cols : list (default=None)
            List of column names to be included.
            If None all columns will be used.
        """
        cols = cols or self.data.keys()
        data = dict((col, str(self.data[col])) for col in cols)
        return data

    def pprint(self, file=None):
        """Pretty-print source data"""
        if not file:
            file = sys.stdout

        pprint(self.data, stream=file)


class SourceCatalog(object):
    """Generic source catalog.

    This class can be used directly, but it's mostly used as a
    base class for the other source catalog classes.

    This is a thin wrapper around `~astropy.table.Table`,
    which is stored in the `catalog.table` attribute.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with catalog data.
    source_name_key : str ('Source_Name')
        Column with source name information
    """
    source_object_class = SourceCatalogObject

    # TODO: at the moment these are duplicated in SourceCatalogObject.
    # Should we share them somehow?
    _source_index_key = 'catalog_row_index'

    def __init__(self, table, source_name_key='Source_Name'):
        self.table = table
        self._source_name_key = source_name_key

        # Make a dict for quick lookup: source name -> row index
        names = dict()
        source_name_col = self.table[self._source_name_key]
        for index, name in enumerate(source_name_col):
            name = name.strip()
            names[name] = index
        self._name_to_index_cache = names

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
        return self._name_to_index_cache[name]

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
        elif isinstance(key, six.integer_types):
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

    def _make_source_dict(self, index):
        """Make one source data dict.

        Parameters
        ----------
        index : int
            Row index

        Returns
        -------
        data : dict
            Source data dict
        """
        row = self.table[index]
        data = OrderedDict(zip(row.colnames, row.as_void()))
        data[self._source_index_key] = index
        return data

    def to_json(self, sources=None, cols=None, format='objects'):
        """Export catalog data to JSON.

        The main use case for this is to display catalog data on webpages,
        where Python / Gammapy and FITS aren't available.

        The formats we support are the ones that can be easily read by
        https://www.datatables.net/manual/data/

        - 'arrays' -- List of lists (row data represented as list)
        - 'objects' -- List of dicts (row data represented as dict)


        Parameters
        ----------
        sources : list, optional
            List of sources names to be included. If it is set to None (default)
            all sources are included.
        cols : list, optional
            List of column names to be included. If it is set to None (default)
            all sources are included.
        format : {'objects', 'arrays'}
            Format specifier (see above, default is 'objects')

        """
        sources = sources or self.table[self._source_name_key]
        cols = cols or self.table.colnames
        
        data = []

        if format == 'arrays':
            for source in sources:
                source = self[source]
                data.append(list(source.data.values()))
        elif format == 'objects':
            for source in sources:
                source = self[source]
                d = source.to_dict(cols=cols)
                data.append(d)
        else:
            raise ValueError('Not a valid data format.')

        return dict(columns=cols, data=data)


# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from collections import OrderedDict
from astropy.table import Table

__all__ = [
    'source_catalogs',
    'SourceCatalogRegistry',
]


class SourceCatalogRegistry(object):
    """Source catalog registry.

    Provides simple and efficient access to source catalogs
    by lazy-loading and caching catalog objects.

    You should use these catalogs read-only, if you modify
    them you can get non-reproducible results if you access
    the modified version later on.
    """

    def __init__(self):
        self._available_catalogs = OrderedDict()
        self._loaded_catalogs = OrderedDict()

    @classmethod
    def builtins(cls):
        """Factory function that make a catalog registry and
        registers the built-in catalogs.
        """
        source_catalogs = cls()

        from .fermi import SourceCatalog3FGL
        source_catalogs.register('3fgl', SourceCatalog3FGL)

        from .fermi import SourceCatalog2FHL
        source_catalogs.register('2fhl', SourceCatalog2FHL)

        return source_catalogs

    @property
    def catalog_names(self):
        return list(self._available_catalogs.keys())

    def register(self, name, factory, args=()):
        """Register a source catalog.

        It must be possible to load it via ``factory(*args)``.
        """
        data = dict(factory=factory, args=args)
        self._available_catalogs[name] = data

    def __getitem__(self, name):
        if name not in self._available_catalogs:
            msg = 'Unknown catalog: "{}". '.format(name)
            msg += 'Available catalogs: {}'.format(self.catalog_names)
            raise KeyError(msg)

        if name not in self._loaded_catalogs:
            cat = self._available_catalogs[name]
            factory = cat['factory']
            args = cat['args']
            self._loaded_catalogs[name] = factory(*args)

        return self._loaded_catalogs[name]

    def info(self, file=None):
        """Print summary info about catalogs.
        """
        if not file:
            file = sys.stdout

        print('Source catalog registry:', file=file)
        # TODO: how can we print to file?
        self.info_table.pprint()

    @property
    def info_table(self):
        """Summary info table on catalogs."""
        table = []
        for name in self._available_catalogs.keys():
            data = dict()
            data['Name'] = name
            data['Description'] = 'description'
            loaded = 'yes' if name in self._loaded_catalogs else 'no'
            data['Loaded'] = loaded
            table.append(data)
        table = Table(rows=table, names=['Name', 'Description', 'Loaded'])
        return table


source_catalogs = SourceCatalogRegistry.builtins()
"""Registry of built-in catalogs in Gammapy.

The main point of the registry is to have one point that
knows about all available catalogs and there's an easy way
to load them.
"""

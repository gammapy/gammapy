# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table

__all__ = ["source_catalogs", "SourceCatalogRegistry"]


class SourceCatalogRegistry(object):
    """Source catalog registry.

    Provides simple and efficient access to source catalogs
    by lazy-loading and caching catalog objects.

    You should use these catalogs read-only, if you modify
    them you can get non-reproducible results if you access
    the modified version later on.
    """

    def __init__(self):
        self._available_catalogs = {}
        self._loaded_catalogs = {}

    @classmethod
    def builtins(cls):
        """Factory function that make a catalog registry and
        registers the built-in catalogs.
        """
        source_catalogs = cls()

        import os

        if "HGPS_ANALYSIS" in os.environ:
            from .hess import SourceCatalogHGPS

            source_catalogs.register("hgps", SourceCatalogHGPS)

        if "GAMMA_CAT" in os.environ:
            from .gammacat import SourceCatalogGammaCat

            source_catalogs.register("gamma-cat", SourceCatalogGammaCat)

        from .fermi import SourceCatalog3FGL

        source_catalogs.register("3fgl", SourceCatalog3FGL)

        from .fermi import SourceCatalog1FHL

        source_catalogs.register("1fhl", SourceCatalog1FHL)

        from .fermi import SourceCatalog2FHL

        source_catalogs.register("2fhl", SourceCatalog2FHL)

        from .fermi import SourceCatalog3FHL

        source_catalogs.register("3fhl", SourceCatalog3FHL)

        from .hawc import SourceCatalog2HWC

        source_catalogs.register("2hwc", SourceCatalog2HWC)

        return source_catalogs

    @property
    def catalog_names(self):
        """Catalog names (list of str)."""
        return list(self._available_catalogs.keys())

    def register(self, name, cls, args=()):
        """Register a source catalog.

        It must be possible to load it via ``cls(*args)``.
        """
        self._available_catalogs[name] = {"cls": cls, "args": args}

    def __getitem__(self, name):
        if name not in self._available_catalogs:
            msg = 'Unknown catalog: "{}". '.format(name)
            msg += "Available catalogs: {}".format(self.catalog_names)
            raise KeyError(msg)

        if name not in self._loaded_catalogs:
            cat = self._available_catalogs[name]
            self._loaded_catalogs[name] = cat["cls"](*cat["args"])

        return self._loaded_catalogs[name]

    def info(self):
        """Print summary info about catalogs."""
        print("Source catalog registry:")
        self.info_table.pprint()

    @property
    def info_table(self):
        """Summary info table on catalogs.

        Loads all catalogs.
        """
        rows = []
        for name in self._available_catalogs.keys():
            cat = self[name]
            rows.append(
                {
                    "name": name,
                    "description": cat.description,
                    "sources": len(cat.table),
                }
            )

        return Table(rows=rows, names=["name", "description", "sources"])


source_catalogs = SourceCatalogRegistry.builtins()
"""Registry of built-in catalogs in Gammapy.

The main point of the registry is to have one point that
knows about all available catalogs and there's an easy way
to load them.
"""

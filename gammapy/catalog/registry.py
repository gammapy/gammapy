# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table
from .fermi import (
    SourceCatalog1FHL,
    SourceCatalog2FHL,
    SourceCatalog3FGL,
    SourceCatalog3FHL,
    SourceCatalog4FGL,
)
from .gammacat import SourceCatalogGammaCat
from .hawc import SourceCatalog2HWC
from .hess import SourceCatalogHGPS

__all__ = ["source_catalogs", "SourceCatalogRegistry"]


class SourceCatalogRegistry:
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
        """Create a catalog registry containing the built-in catalogs."""
        cats = cls()
        cats.register("hgps", SourceCatalogHGPS)
        cats.register("gamma-cat", SourceCatalogGammaCat)
        cats.register("3fgl", SourceCatalog3FGL)
        cats.register("4fgl", SourceCatalog4FGL)
        cats.register("1fhl", SourceCatalog1FHL)
        cats.register("2fhl", SourceCatalog2FHL)
        cats.register("3fhl", SourceCatalog3FHL)
        cats.register("2hwc", SourceCatalog2HWC)
        return cats

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
            msg = f"Unknown catalog: {name!r}. "
            msg += f"Available catalogs: {self.catalog_names!r}"
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
"""Registry of built-in catalogs in Gammapy."""

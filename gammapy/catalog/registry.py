# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.utils.registry import Registry
from .fermi import (
    SourceCatalog2FHL,
    SourceCatalog3FGL,
    SourceCatalog3FHL,
    SourceCatalog4FGL,
)
from .gammacat import SourceCatalogGammaCat
from .hawc import SourceCatalog2HWC
from .hess import SourceCatalogHGPS

__all__ = ["SOURCE_CATALOGS"]

SOURCE_CATALOGS = Registry([
    SourceCatalogGammaCat,
    SourceCatalogHGPS,
    SourceCatalog2HWC,
    SourceCatalog3FGL,
    SourceCatalog4FGL,
    SourceCatalog2FHL,
    SourceCatalog3FHL
])
"""Registry of source catalogs in Gammapy."""

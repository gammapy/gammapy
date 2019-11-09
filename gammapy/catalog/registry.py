# Licensed under a 3-clause BSD style license - see LICENSE.rst
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

SOURCE_CATALOGS = {
    "gamma-cat": SourceCatalogGammaCat,
    "hgps": SourceCatalogHGPS,
    "2hwc": SourceCatalog2HWC,
    "3fgl": SourceCatalog3FGL,
    "4fgl": SourceCatalog4FGL,
    "2fhl": SourceCatalog2FHL,
    "3fhl": SourceCatalog3FHL,
}
"""Registry of source catalogs in Gammapy."""

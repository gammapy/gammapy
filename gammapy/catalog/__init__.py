# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalogs."""
from gammapy.utils.registry import Registry
from .core import *
from .fermi import *
from .gammacat import *
from .hawc import *
from .hess import *

CATALOG_REGISTRY = Registry(
    [
        SourceCatalogGammaCat,
        SourceCatalogHGPS,
        SourceCatalog2HWC,
        SourceCatalog3FGL,
        SourceCatalog4FGL,
        SourceCatalog2FHL,
        SourceCatalog3FHL,
        SourceCatalog3HWC,
    ]
)
"""Registry of source catalogs in Gammapy."""


__all__ = [
    "CATALOG_REGISTRY",
    "SourceCatalog",
    "SourceCatalogObjectHGPS",
    "SourceCatalogObject2FHL",
    "SourceCatalogObject3FHL",
    "SourceCatalogObject3FGL",
    "SourceCatalogObject4FGL",
    "SourceCatalogObject2HWC",
    "SourceCatalogObject3HWC",
    "SourceCatalogObjectGammaCat",
    "SourceCatalogObjectHGPSComponent",
    "SourceCatalogLargeScaleHGPS",
    "SourceCatalogObject",
]

__all__.extend(cls.__name__ for cls in CATALOG_REGISTRY)

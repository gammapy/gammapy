# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalogs."""
from gammapy.utils.registry import Registry
from .core import SourceCatalog, SourceCatalogObject
from .fermi import (
    SourceCatalog2FHL,
    SourceCatalog2PC,
    SourceCatalog3FGL,
    SourceCatalog3FHL,
    SourceCatalog3PC,
    SourceCatalog4FGL,
    SourceCatalogObject2FHL,
    SourceCatalogObject2PC,
    SourceCatalogObject3FGL,
    SourceCatalogObject3FHL,
    SourceCatalogObject3PC,
    SourceCatalogObject4FGL,
)
from .gammacat import SourceCatalogGammaCat, SourceCatalogObjectGammaCat
from .hawc import (
    SourceCatalog2HWC,
    SourceCatalog3HWC,
    SourceCatalogObject2HWC,
    SourceCatalogObject3HWC,
)
from .hess import (
    SourceCatalogHGPS,
    SourceCatalogLargeScaleHGPS,
    SourceCatalogObjectHGPS,
    SourceCatalogObjectHGPSComponent,
)
from .lhaaso import SourceCatalog1LHAASO, SourceCatalogObject1LHAASO

CATALOG_REGISTRY = Registry(
    [
        SourceCatalogGammaCat,
        SourceCatalogHGPS,
        SourceCatalog2HWC,
        SourceCatalog3FGL,
        SourceCatalog4FGL,
        SourceCatalog2FHL,
        SourceCatalog3FHL,
        SourceCatalog3PC,
        SourceCatalog3HWC,
        SourceCatalog2PC,
        SourceCatalog1LHAASO,
    ]
)
"""Registry of source catalogs in Gammapy."""


__all__ = [
    "CATALOG_REGISTRY",
    "SourceCatalog",
    "SourceCatalog1LHAASO",
    "SourceCatalog2FHL",
    "SourceCatalog2HWC",
    "SourceCatalog3FGL",
    "SourceCatalog3FHL",
    "SourceCatalog3HWC",
    "SourceCatalog4FGL",
    "SourceCatalog2PC",
    "SourceCatalog3PC",
    "SourceCatalogGammaCat",
    "SourceCatalogHGPS",
    "SourceCatalogLargeScaleHGPS",
    "SourceCatalogObject",
    "SourceCatalogObject1LHAASO",
    "SourceCatalogObject2FHL",
    "SourceCatalogObject2HWC",
    "SourceCatalogObject3FGL",
    "SourceCatalogObject3FHL",
    "SourceCatalogObject3HWC",
    "SourceCatalogObject4FGL",
    "SourceCatalogObject2PC",
    "SourceCatalogObject3PC",
    "SourceCatalogObjectGammaCat",
    "SourceCatalogObjectHGPS",
    "SourceCatalogObjectHGPSComponent",
]

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module builds all of the ASDF extensions which will be registered by `gammapy.io.asdf.integration`,
via an ``entry-point`` in the ``pyproject.toml`` file.
"""

from asdf.extension import ManifestExtension

from .converters.data.gti import GTIConverter
from .converters.maps.axes import (
    LabelMapAxisConverter,
    MapAxesConverter,
    MapAxisConverter,
    TimeMapAxisConverter,
)
from .converters.maps.geom import (
    HpxGeomConverter,
    RegionGeomConverter,
    WcsGeomConverter,
)

from .converters.maps.maps import MapsConverter
from .converters.maps.ndmap import (
    HpxNDMapConverter,
    RegionNDMapConverter,
    WcsNDMapConverter,
)

from .converters.irf.psf.map import PSFMapConverter, RecoPSFMapConverter
from .converters.irf.edisp.map import EDispMapConverter, EDispKernelMapConverter

GAMMAPY_CONVERTERS = [
    MapsConverter(),
    GTIConverter(),
    MapAxisConverter(),
    MapAxesConverter(),
    TimeMapAxisConverter(),
    LabelMapAxisConverter(),
    HpxGeomConverter(),
    RegionGeomConverter(),
    WcsGeomConverter(),
    HpxNDMapConverter(),
    RegionNDMapConverter(),
    WcsNDMapConverter(),
    PSFMapConverter(),
    RecoPSFMapConverter(),
    EDispMapConverter(),
    EDispKernelMapConverter(),
]

GAMMAPY_EXTENSIONS = [
    ManifestExtension.from_uri(
        "asdf://gammapy.org/gammapy/manifests/gammapy-1.0.0",
        converters=GAMMAPY_CONVERTERS,
    )
]

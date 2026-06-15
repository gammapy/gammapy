# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module builds all of the ASDF extensions which will be registered by `gammapy.io.asdf.integration`,
via an ``entry-point`` in the ``pyproject.toml`` file.
"""

from asdf.extension import ManifestExtension

from .converters.maps.axes import (
    MapAxisConverter,
    TimeMapAxisConverter,
    LabelMapAxisConverter,
)

GAMMAPY_CONVERTERS = [
    MapAxisConverter(),
    TimeMapAxisConverter(),
    LabelMapAxisConverter(),
]

GAMMAPY_EXTENSIONS = [
    ManifestExtension.from_uri(
        "asdf://gammapy.org/gammapy/manifests/gammapy-1.0.0",
        converters=GAMMAPY_CONVERTERS,
    )
]

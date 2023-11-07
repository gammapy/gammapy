# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.makers import MAKER_REGISTRY


def test_maker_registry():
    assert "Maker" in str(MAKER_REGISTRY)

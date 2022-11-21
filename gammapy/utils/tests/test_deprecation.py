# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ..deprecation import GammapyDeprecationWarning, gammapy_deprecated


@gammapy_deprecated("v1.1", alternative="new_function")
def deprecated_function(a, b):
    return a + b


@gammapy_deprecated(since="v1.2")
class deprecated_class:
    def __init__(self):
        pass


def test_deprecated_function():
    assert "deprecated:: v1.1" in deprecated_function.__doc__

    with pytest.warns(GammapyDeprecationWarning, match="Use new_function instead"):
        deprecated_function(1, 2)


def test_deprecated_class():
    assert "deprecated:: v1.2" in deprecated_class.__doc__

    with pytest.warns(
        GammapyDeprecationWarning, match="The deprecated_class class is deprecated"
    ):
        deprecated_class()

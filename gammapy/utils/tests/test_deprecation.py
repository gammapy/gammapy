# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ..deprecation import (
    GammapyDeprecationWarning,
    gammapy_deprecated,
    gammapy_deprecated_renamed_argument,
)


@gammapy_deprecated("v1.1", alternative="new_function")
def deprecated_function(a, b):
    return a + b


@gammapy_deprecated(since="v1.2")
class deprecated_class:
    def __init__(self):
        pass


@gammapy_deprecated_renamed_argument("a", "x", "v1.1")
def deprecated_argument_function(x, y):
    return x + y


@gammapy_deprecated_renamed_argument("old", "new", "v1.1", arg_in_kwargs=True)
def deprecated_argument_function_kwarg(new=1):
    return new


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


def test_deprecated_argument():
    with pytest.warns(GammapyDeprecationWarning):
        res = deprecated_argument_function(a=1, y=2)
        assert res == 3

    with pytest.raises(TypeError):
        deprecated_argument_function(a=1, x=2, y=2)

    with pytest.warns(GammapyDeprecationWarning):
        res = deprecated_argument_function_kwarg(old=2)
        assert res == 2

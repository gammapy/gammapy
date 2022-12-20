# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ..deprecation import (
    GammapyDeprecationWarning,
    deprecated,
    deprecated_attribute,
    deprecated_renamed_argument,
)


@deprecated("v1.1", alternative="new_function")
def deprecated_function(a, b):
    return a + b


@deprecated(since="v1.2")
class deprecated_class:
    def __init__(self):
        pass


@deprecated_renamed_argument("a", "x", "v1.1")
def deprecated_argument_function(x, y):
    return x + y


@deprecated_renamed_argument("old", "new", "v1.1", arg_in_kwargs=True)
def deprecated_argument_function_kwarg(new=1):
    return new


class some_class:
    old_attribute = deprecated_attribute(
        "old_attribute", "v1.1", alternative="new_attribute"
    )

    def __init__(self, value):
        self._old_attribute = value
        self._new_attribute = value

    @property
    def new_attribute(self):
        return self._new_attribute


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

    # this warns first and then raises
    with pytest.warns(GammapyDeprecationWarning):
        with pytest.raises(TypeError):
            deprecated_argument_function(a=1, x=2, y=2)

    with pytest.warns(GammapyDeprecationWarning):
        res = deprecated_argument_function_kwarg(old=2)
        assert res == 2


def test_deprecated_attibute():
    object = some_class(1)
    with pytest.warns(GammapyDeprecationWarning):
        res = object.old_attribute
        assert res == 1

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for caching"""

# from https://github.com/python/cpython/issues/102618#issuecomment-2839489762.
import functools
import inspect
import weakref
from typing import ParamSpec, TypeVar

_Self = TypeVar("_Self")
_Params = ParamSpec("_Params")
_Return = TypeVar("_Return")


class _WeakIdDict:
    """Like `weakref.WeakKeyDictionary`, but uses identity-based hashing and equality."""

    __slots__ = ("_dict",)

    def __init__(self):
        self._dict = {}

    def __getitem__(self, key):
        item, _ = self._dict[id(key)]
        return item

    def __setitem__(self, key, value):
        id_key = id(key)
        ref = weakref.ref(key, lambda _: self._dict.pop(id_key))
        self._dict[id_key] = (value, ref)


class _CacheKey:
    "Cache key used to compute an object's hash just once."

    __slots__ = ("hashvalue", "value")

    def __init__(self, value):
        self.hashvalue = hash(value)
        self.value = value

    def __hash__(self) -> int:
        return self.hashvalue

    def __eq__(self, other) -> bool:
        # Assume `type(other) is _Key`
        return self.value == other.value


def cachemethod(fn):
    """Like `functools.cache`, except that it only holds a weak reference to its first argument.

    Note that `functools.cached_property` (which also uses a weak reference) can often be used for similar purposes.
    The differences are that:
        (a) `cached_property` will be pickled while `cachemethod` will not,
        (b) `cachemethod`can be used on functions with additional arguments,
        (c) `cachemethod` requires brackets to call, helping to visually emphasise that computationl work may be being performed.
    """
    cache1 = _WeakIdDict()
    sig = inspect.signature(fn)
    parameters = list(sig.parameters.values())
    if len(parameters) == 0:
        raise ValueError(
            "Cannot use `cachemethod` on a function without a `self` argument."
        )
    if parameters[0].kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise ValueError(
            "Cannot use `cachemethod` on a function without a positional argument."
        )
    parameters = parameters[1:]
    sig = sig.replace(parameters=parameters)

    @ functools.wraps(fn)
    def fn_wrapped(self, *args, **kwargs):
        # Standardise arguments to a single form to encourage cache hits.
        # Not binding `self` (and instead doing the signature-adjustment above) in order to avoid keeping a strong
        # reference to `self` via `argkey`.
        bound = sig.bind(*args, **kwargs)
        del args, kwargs
        argkey = _CacheKey((bound.args, tuple(bound.kwargs.items())))
        try:
            out = cache1[self][argkey]
        except KeyError:
            try:
                cache2 = cache1[self]
            except KeyError:
                cache2 = cache1[self] = {}
            out = cache2[argkey] = fn(self, *bound.args, **bound.kwargs)
        return out

    return fn_wrapped

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for caching"""

import functools
import inspect
import hashlib
import weakref
from gammapy.utils.parallel import is_ray_available

USE_INSTANCE_CACHE = False

if is_ray_available():
    import ray

    def _hash(value):
        try:
            return hash(value)
        except TypeError:
            data = ray.cloudpickle.dumps(value)
            return hashlib.sha256(data).hexdigest()
else:

    def _hash(value):
        return hash(value)


def make_key(sig, *args, **kwargs):
    """
    Generate a unique hash key for caching based on normalized constructor arguments.

    This function uses the signature of the class constructor (`__init__`) to normalize
    the input arguments, serializes them using `ray.cloudpickle`, and returns a SHA-256 hash
    digest. The resulting key is suitable for use in caching mechanisms.

    Parameters
    ----------
    sig : inspect.Signature
        The signature of the method or function used to normalize arguments.
    *args : tuple
        Positional arguments to be normalized.
    **kwargs : dict
        Keyword arguments to be normalized.

    Returns
    -------
    key : str
        A SHA-256 hexadecimal digest representing the normalized arguments.
    """
    bound = sig.bind_partial(None, *args, **kwargs)
    bound.apply_defaults()
    normalized_args = dict(sorted(bound.arguments.items()))
    normalized_args.pop("self", None)
    return _hash(tuple(normalized_args.items()))


class _WeakIdDict:
    """Like `weakref.WeakKeyDictionary`, but uses identity-based hashing and equality.
    from https://github.com/python/cpython/issues/102618#issuecomment-2839489762.
    """

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


def cachemethod(fn):
    """
    Decorator to cache method results on a per-instance basis using weak references.

    This decorator stores cached results of method calls in a `WeakKeyDictionary`,
    ensuring that the cache is automatically cleared when the instance is garbage collected.
    The cache key is generated using `make_key`, which normalizes and hashes the method
    arguments to ensure consistent and order-independent caching.

    Parameters
    ----------
    fn : callable
        The method to be decorated.

    Returns
    -------
    wrapper : callable
        The wrapped method with caching behavior.
    """

    cache1 = _WeakIdDict()
    sig = inspect.signature(fn)

    parameters = list(sig.parameters.values())
    if len(parameters) == 0 or parameters[0].kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise ValueError(
            "The `@cachemethod` decorator can only be used on instance methods "
            "with a positional `self` argument."
        )

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        argkey = make_key(sig, *args, **kwargs)
        try:
            out = cache1[self][argkey]
        except KeyError:
            try:
                cache2 = cache1[self]
            except KeyError:
                cache2 = cache1[self] = {}
            out = cache2[argkey] = fn(self, *args, **kwargs)
        return out

    return wrapper


class CacheInstanceMixin:
    """Cache class instance based on init arguments equivalence"""

    def __new__(cls, *args, **kwargs):
        new = super().__new__(cls)
        if not USE_INSTANCE_CACHE or not is_ray_available():
            return new

        # Ensure each subclass has its own cache
        if not hasattr(cls, "_instances"):
            cls._instances = _WeakIdDict()

        try:
            new.__init__(*args, **kwargs)
        except TypeError:
            return new  # ignore cache for unpickle

        sig = inspect.signature(cls.__init__)
        argkey = make_key(sig, *args, **kwargs)
        try:
            out = cls._instances[cls][argkey]
        except KeyError:
            try:
                cache2 = cls._instances[cls]
            except KeyError:
                cache2 = cls._instances[cls] = {}
            out = cache2[argkey] = new
        return out

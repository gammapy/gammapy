# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for caching"""

import cloudpickle
import hashlib
import inspect
import weakref


def make_key(cls, *args, **kwargs):
    # Normalize arguments using the signature of __init__
    sig = inspect.signature(cls.__init__)
    bound = sig.bind_partial(None, *args, **kwargs)  # 'None' for 'self'
    bound.apply_defaults()

    # Sort arguments by name to ensure order-independence
    normalized_args = dict(sorted(bound.arguments.items()))
    # Remove 'self' if present
    normalized_args.pop("self", None)

    # Serialize and hash compatible with pickle
    data = cloudpickle.dumps(normalized_args)
    return hashlib.sha256(data).hexdigest()


class CacheEquivalentMixin:
    """Cache class instance"""

    def __new__(cls, *args, **kwargs):
        # Ensure each subclass has its own cache
        if not hasattr(cls, "_instances"):
            cls._instances = weakref.WeakValueDictionary()

        key = make_key(cls, *args, **kwargs)
        if key in cls._instances:
            return cls._instances[key]
        instance = super().__new__(cls)
        cls._instances[key] = instance
        return instance

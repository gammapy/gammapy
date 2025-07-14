# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for caching"""

import weakref


class CacheEquivalentMixin:
    """Cache class instance using equality condition.
    Simpler for classes with complex arguements that are not hashable.
    """

    _instances = []

    def __new__(cls, *args, **kwargs):
        # Clean up dead references
        cls._instances = [ref for ref in cls._instances if ref() is not None]

        # Create a temporary instance to compare
        temp = super().__new__(cls)
        try:
            temp.__init__(*args, **kwargs)
        except TypeError:
            return temp  # ignore cache for unpickle

        # Search for an equivalent instance
        for ref in cls._instances:
            instance = ref()
            if instance is not None and instance == temp:
                return instance

        # Store a weak reference to the new instance
        cls._instances.append(weakref.ref(temp))
        return temp

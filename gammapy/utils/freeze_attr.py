# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Decorator to freeze class attribute setting."""

from functools import wraps


def freeze(cls):
    """Decorator function to freeze public instance attributes after calling its __init__() method.

    The list of allowed attributes is obtained from the instance level attributes and the class level attributes
    by the set_allowed_attrs function. It only considers public attributes (i.e. without _ at the start).

    The __setattr__ method is overridden to check if the key is in the list of allowed attributes.

    A ._frozen attribute is set to True only by the __init__() of the decorated function to support inheritance
    of frozen classes.
    """
    original_init = cls.__init__

    def find_allowed_attrs(obj):
        instance_attrs = {k for k in obj.__dict__ if not k.startswith("_")}

        # Public class-level attributes (excluding callables, but include properties with setters)
        class_attrs = set()
        for base in obj.__class__.__mro__:
            for name, value in vars(base).items():
                if name.startswith("_"):
                    continue
                if isinstance(value, property):
                    if value.fset is not None:
                        class_attrs.add(name)
                elif not callable(value):
                    class_attrs.add(name)

        return instance_attrs | class_attrs

    @wraps(original_init)
    def __init__(self, *args, **kwargs):
        self._frozen_attrs = False

        # Call the original constructor
        original_init(self, *args, **kwargs)

        # Freeze the object only if we are in its __init__ method
        if self.__class__ is cls:
            self._allowed_attrs = find_allowed_attrs(self)
            self._frozen_attrs = True

    def __setattr__(self, key, value):
        if (
            getattr(self, "_frozen", False)
            and not key.startswith("_")
            and key not in getattr(self, "_allowed_attrs", set())
        ):
            raise AttributeError(
                f"{key} is not a valid attribute of {type(self).__name__}."
            )
        object.__setattr__(self, key, value)

    cls.__init__ = __init__
    cls.__setattr__ = __setattr__

    return cls

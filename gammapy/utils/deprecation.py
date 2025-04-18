# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = [
    "GammapyDeprecationWarning",
    "deprecated",
    "deprecated_renamed_argument",
    "deprecated_attribute",
]


class GammapyDeprecationWarning(Warning):
    """The Gammapy deprecation warning."""


def deprecated(since, **kwargs):
    """
    Use to mark a function or class as deprecated.

    Reuses Astropy's deprecated decorator.
    Check arguments and usage in `~astropy.utils.decorator.deprecated`.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated. This is required.
    """
    from astropy.utils import deprecated

    kwargs["warning_type"] = GammapyDeprecationWarning
    return deprecated(since, **kwargs)


def deprecated_renamed_argument(old_name, new_name, since, **kwargs):
    """Deprecate a _renamed_ or _removed_ function argument.

    Check arguments and usage in `~astropy.utils.decorator.deprecated_renamed_argument`.
    """
    from astropy.utils import deprecated_renamed_argument

    kwargs["warning_type"] = GammapyDeprecationWarning
    return deprecated_renamed_argument(old_name, new_name, since, **kwargs)


def deprecated_attribute(name, since, **kwargs):
    """
    Use to mark a public attribute as deprecated.

    This creates a property that will warn when the given attribute name is accessed.
    """
    from astropy.utils import deprecated_attribute

    kwargs["warning_type"] = GammapyDeprecationWarning
    return deprecated_attribute(name, since, **kwargs)


def deprecated_key(key, old_keys, new_keys, since):
    """
    Raise a deprecation warning if provided key is deprecated.

    Returns the corresponding new key if the given is deprecated.
    """
    import warnings

    if key in old_keys:
        index = old_keys.index(key)
        new_key = new_keys[index]
        warnings.warn(
            f'"{key}" was deprecated in version {since} and will be removed in a future version. '
            f'Use "{new_key}" instead.',
            GammapyDeprecationWarning,
            stacklevel=2,
        )
        return new_key
    return key


class DeprecatedDict(dict):
    """
    Raise a deprecation warning if provided key is deprecated.

    Returns the corresponding new key if the given is deprecated.
    """

    def __init__(self, data, old_keys_to_new_keys, since):
        super().__init__(data)
        self._old_keys_to_new_keys = old_keys_to_new_keys or {}
        self._since = since

    def _warn_and_update_key(self, key):
        """Helper function to handle deprecated key warnings and return the new key."""
        import warnings

        if key in self._old_keys_to_new_keys:
            new_key = self._old_keys_to_new_keys[key]
            warnings.warn(
                f'"{key}" was deprecated in version {self._since} and will be removed in a future version. '
                f'Use "{new_key}" instead.',
                GammapyDeprecationWarning,
            )
            return new_key
        return key

    def __getitem__(self, key):
        key = self._warn_and_update_key(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        key = self._warn_and_update_key(key)
        return super().__setitem__(key, value)

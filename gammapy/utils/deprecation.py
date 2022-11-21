# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.utils import deprecated, deprecated_renamed_argument

__all__ = [
    "GammapyDeprecationWarning",
    "gammapy_deprecated",
    "gammapy_deprecated_renamed_argument",
]


class GammapyDeprecationWarning(Warning):
    """
    The Gammapy deprecation warning
    """


def gammapy_deprecated(since, **kwargs):
    """
    Used to mark a function or class as deprecated.

    Reuses Astropy's deprecated decorator.
    Check arguments and usage in `~astropy.utils.decorator.deprecated`

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.  This is required.
    """
    kwargs["warning_type"] = GammapyDeprecationWarning
    return deprecated(since, **kwargs)


def gammapy_deprecated_renamed_argument(old_name, new_name, since, **kwargs):
    """Deprecate a _renamed_ or _removed_ function argument.

    Check arguments and usage in `~astropy.utils.decorator.deprecated_renamed_argument`
    """
    kwargs["warning_type"] = GammapyDeprecationWarning
    return deprecated_renamed_argument(old_name, new_name, since, **kwargs)

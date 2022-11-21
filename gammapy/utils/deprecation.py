# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.utils import deprecated

__all__ = ["GammapyDeprecationWarning", "gammapy_deprecated"]


class GammapyDeprecationWarning(Warning):
    """
    The Gammapy deprecation warning
    """


def gammapy_deprecated(
    since,
    message="",
    name="",
    alternative="",
    pending=False,
    obj_type=None,
    warning_type=GammapyDeprecationWarning,
):
    """
    Used to mark a function or class as deprecated.

    Reuses Astropy's deprecated decorator

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.  This is required.

    message : str, optional
        Override the default deprecation message.  The format
        specifier ``func`` may be used for the name of the function,
        and ``alternative`` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        function. ``obj_type`` may be used to insert a friendly name
        for the type of object being deprecated.

    name : str, optional
        The name of the deprecated function or class; if not provided
        the name is automatically determined from the passed in
        function or class, though this is useful in the case of
        renamed functions, where the new function is just assigned to
        the name of the deprecated function.  For example::

            def new_function():
                ...
            oldFunction = new_function

    alternative : str, optional
        An alternative function or class name that the user may use in
        place of the deprecated object.  The deprecation warning will
        tell the user about this alternative if provided.

    obj_type : str, optional
        The type of this object, if the automatically determined one
        needs to be overridden.

    warning_type : Warning
        Warning to be issued.
        Default is `~gammapy.utils.GammapyDeprecationWarning`.
    """
    return deprecated(
        since,
        message=message,
        name=name,
        alternative=alternative,
        pending=pending,
        obj_type=obj_type,
        warning_type=warning_type,
    )

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import matplotlib

__all__ = ["BasePlotter"]


class BasePlotter:
    """Base class for Gammapy plotter objects.

    A plotter carries a local matplotlib configuration and exposes plotting
    methods that receive the object to plot as an argument. It keeps no
    reference to a data object.

    Parameters
    ----------
    rc_params : dict, optional
        Mapping of `matplotlib.rcParams` keys to values. Entries are validated
        when set and applied while plotting. Default is None.
    """

    def __init__(self, rc_params=None):
        self._rc_params = matplotlib.RcParams()
        if rc_params is not None:
            self.rc_params = rc_params

    @property
    def rc_params(self):
        """Local matplotlib configuration as a `~matplotlib.RcParams` object."""
        return self._rc_params

    @rc_params.setter
    def rc_params(self, value):
        if not isinstance(value, dict):
            raise TypeError("rc_params must be a dictionary")

        merged = matplotlib.RcParams(self._rc_params)
        merged.update(value)
        self._rc_params = merged

    def _rc_context(self):
        """Context manager applying the local configuration while plotting."""
        return matplotlib.rc_context(self._rc_params)

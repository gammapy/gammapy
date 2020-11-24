# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import inspect
from copy import deepcopy
import numpy as np
from gammapy.modeling.models import Model

__all__ = ["Estimator"]


class Estimator(abc.ABC):
    """Abstract estimator base class."""

    _available_selection_optional = {}

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def run(self, datasets):
        pass

    @property
    def selection_optional(self):
        """"""
        return self._selection_optional

    @selection_optional.setter
    def selection_optional(self, selection):
        """Set optional selection"""
        available = self._available_selection_optional

        if selection == "all":
            self._selection_optional = available
        elif selection is None:
            self._selection_optional = []
        else:
            if set(selection).issubset(set(available)):
                self._selection_optional = selection
            else:
                difference = set(selection).difference(set(available))
                raise ValueError(f"{difference} is not a valid method.")

    @staticmethod
    def get_sqrt_ts(ts, norm):
        r"""Compute sqrt(TS) value.

        Compute sqrt(TS) as defined by:

        .. math::
            \sqrt{TS} = \left \{
            \begin{array}{ll}
              -\sqrt{TS} & : \text{if} \ norm < 0 \\
              \sqrt{TS} & : \text{else}
            \end{array}
            \right.

        Parameters
        ----------
        ts : `~numpy.ndarray`
            TS value.
        norm : `~numpy.ndarray`
            norm value
        Returns
        -------
        sqrt_ts : `~numpy.ndarray`
            Sqrt(TS) value.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(norm > 0, np.sqrt(ts), -np.sqrt(ts))

    def copy(self):
        """Copy estimator"""
        return deepcopy(self)

    @property
    def config_parameters(self):
        """Config parameters"""
        pars = {}
        names = self.__init__.__code__.co_varnames
        for name in names:
            if name == "self":
                continue

            pars[name] = getattr(self, name)
        return pars

    def __str__(self):
        s = f"{self.__class__.__name__}\n"
        s += "-" * (len(s) - 1) + "\n\n"

        pars = self.config_parameters
        max_len = np.max([len(_) for _ in pars]) + 1

        for name, value in sorted(pars.items()):
            if isinstance(value, Model):
                s += f"\t{name:{max_len}s}: {value.__class__.__name__}\n"
            elif inspect.isclass(value):
                s += f"\t{name:{max_len}s}: {value.__name__}\n"
            elif isinstance(value, np.ndarray):
                s += f"\t{name:{max_len}s}: {value}\n"
            else:
                s += f"\t{name:{max_len}s}: {value}\n"

        return s.expandtabs(tabsize=2)

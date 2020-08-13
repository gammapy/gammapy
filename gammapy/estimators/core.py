# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
from copy import deepcopy
import numpy as np
from gammapy.modeling.models import Model

__all__ = ["Estimator"]


class Estimator(abc.ABC):
    """Abstract estimator base class."""

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def run(self, datasets):
        pass

    def _make_selection(self, selection):
        if selection == "all":
            return self.available_selection
        elif selection is None:
            return set()
        elif isinstance(selection, str) and selection in self.available_selection:
            return set([selection])
        else:
            selection = set(selection)
            if selection.issubset(self.available_selection):
                return selection
            else:
                raise ValueError(
                    f"Incorrect selection. Available options are {self.available_selection}"
                )

    @staticmethod
    def get_sqrt_ts(ts):
        r"""Compute sqrt(TS) value.

        Compute sqrt(TS) as defined by:

        .. math::
            \sqrt{TS} = \left \{
            \begin{array}{ll}
              -\sqrt{-TS} & : \text{if} \ TS < 0 \\
              \sqrt{TS} & : \text{else}
            \end{array}
            \right.

        Parameters
        ----------
        ts : `~numpy.ndarray`
            TS value.

        Returns
        -------
        sqrt_ts : `~numpy.ndarray`
            Sqrt(TS) value.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ts > 0, np.sqrt(ts), -np.sqrt(-ts))

    # TODO: replace this type checking by using pydantic models in future
    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, selection):
        self._selection = self._make_selection(selection)

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

        for name, value in pars.items():
            if isinstance(value, Model):
                s += f"\t{name:{max_len}s}: {value.__class__.__name__}\n"
            elif isinstance(value, np.ndarray):
                s += f"\t{name:{max_len}s}: {value}\n"
            else:
                s += f"\t{name:{max_len}s}: {value}\n"

        return s.expandtabs(tabsize=2)

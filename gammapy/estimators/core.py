# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import inspect
from copy import deepcopy
import numpy as np
from astropy import units as u
from gammapy.maps import MapAxis
from gammapy.modeling.models import ModelBase

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

        if selection is None:
            self._selection_optional = []
        elif "all" in selection:
            self._selection_optional = available
        else:
            if set(selection).issubset(set(available)):
                self._selection_optional = selection
            else:
                difference = set(selection).difference(set(available))
                raise ValueError(f"{difference} is not a valid method.")

    def _get_energy_axis(self, dataset):
        """Energy axis"""
        if self.energy_edges is None:
            energy_axis = dataset.counts.geom.axes["energy"].squash()
        else:
            energy_axis = MapAxis.from_energy_edges(self.energy_edges)

        return energy_axis

    def copy(self):
        """Copy estimator"""
        return deepcopy(self)

    @property
    def config_parameters(self):
        """Config parameters"""
        pars = self.__dict__.copy()
        pars = {key.strip("_"): value for key, value in pars.items()}
        return pars

    def __str__(self):
        s = f"{self.__class__.__name__}\n"
        s += "-" * (len(s) - 1) + "\n\n"

        pars = self.config_parameters
        max_len = np.max([len(_) for _ in pars]) + 1

        for name, value in sorted(pars.items()):
            if isinstance(value, ModelBase):
                s += f"\t{name:{max_len}s}: {value.tag[0]}\n"
            elif inspect.isclass(value):
                s += f"\t{name:{max_len}s}: {value.__name__}\n"
            elif isinstance(value, u.Quantity):
                s += f"\t{name:{max_len}s}: {value}\n"
            elif isinstance(value, Estimator):
                pass
            elif isinstance(value, np.ndarray):
                value = np.array_str(value, precision=2, suppress_small=True)
                s += f"\t{name:{max_len}s}: {value}\n"
            else:
                s += f"\t{name:{max_len}s}: {value}\n"

        return s.expandtabs(tabsize=2)

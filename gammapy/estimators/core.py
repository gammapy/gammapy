# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc

__all__ = ["Estimator"]


class Estimator(abc.ABC):
    """Abstract estimator base class."""

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

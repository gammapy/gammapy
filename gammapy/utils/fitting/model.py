# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
from .parameter import Parameters

__all__ = ["Model"]


class Model:
    """Model base class."""

    __slots__ = ["_parameters"]

    def __init__(self, params=None):
        self._parameters = Parameters(params)

    @property
    def parameters(self):
        """Parameters (`~gammapy.utils.modeling.Parameters`)"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n\nParameters: \n\n\t"

        table = self.parameters.to_table()
        ss += "\n\t".join(table.pformat())

        if self.parameters.covariance is not None:
            ss += "\n\nCovariance: \n\n\t"
            covariance = self.parameters.covariance_to_table()
            ss += "\n\t".join(covariance.pformat())
        return ss

    def to_dict(self, selection="all"):
        return {
            "type": self.__class__.__name__,
            "parameters": self.parameters.to_dict(selection)["parameters"],
        }

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from gammapy.modeling import PriorParameter, PriorParameters
from .core import ModelBase

log = logging.getLogger(__name__)


def _build_priorparameters_from_dict(data, default_parameters):
    """Build PriorParameters object from input dict and the default prior parameter values from the PriorModel class."""
    par_data = []

    input_names = [_["name"] for _ in data]

    for par in default_parameters:
        par_dict = par.to_dict()
        try:
            index = input_names.index(par_dict["name"])
            par_dict.update(data[index])
        except ValueError:
            log.warning(
                f"PriorParameter '{par_dict['name']}' not defined in YAML file."
                f" Using default value: {par_dict['value']} {par_dict['unit']}"
            )
        par_data.append(par_dict)

    return PriorParameters.from_dict(par_data)


class Prior(ModelBase):
    _unit = ""

    def __init__(self, **kwargs):
        # Copy default parameters from the class to the instance
        default_parameters = self.default_parameters.copy()

        for par in default_parameters:
            value = kwargs.get(par.name, par)
            if not isinstance(value, PriorParameter):
                par.quantity = u.Quantity(value)
            else:
                par = value

            setattr(self, par.name, par)

        _weight = kwargs.get("weight", None)

        if _weight is not None:
            self._weight = _weight
        else:
            self._weight = 1

    @property
    def parameters(self):
        """PriorParameters (`~gammapy.modeling.PriorParameters`)"""
        return PriorParameters(
            [getattr(self, name) for name in self.default_parameters.names]
        )

    def __init_subclass__(cls, **kwargs):
        # Add priorparameters list on the model sub-class (not instances)
        cls.default_parameters = PriorParameters(
            [_ for _ in cls.__dict__.values() if isinstance(_, PriorParameter)]
        )

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    def __call__(self, value):
        """Call evaluate method"""
        # assuming the same unit as the PriorParamter here
        kwargs = {par.name: par.value for par in self.parameters}
        return self.weight * self.evaluate(value.value, **kwargs)

    def to_dict(self, full_output=False):
        """Create dict for YAML serialisation"""
        tag = self.tag[0] if isinstance(self.tag, list) else self.tag
        params = self.parameters.to_dict()

        if not full_output:
            for par, par_default in zip(params, self.default_parameters):
                init = par_default.to_dict()
                for item in [
                    "min",
                    "max",
                    "error",
                ]:
                    default = init[item]

                    if par[item] == default or (
                        np.isnan(par[item]) and np.isnan(default)
                    ):
                        del par[item]

        data = {"type": tag, "parameters": params, "weight": self.weight}

        if self.type is None:
            return data
        else:
            return {self.type: data}

    @classmethod
    def from_dict(cls, data):
        kwargs = {}

        key0 = next(iter(data))
        if key0 in ["prior"]:
            data = data[key0]
        if data["type"] not in cls.tag:
            raise ValueError(
                f"Invalid model type {data['type']} for class {cls.__name__}"
            )

        priorparameters = _build_priorparameters_from_dict(
            data["parameters"], cls.default_parameters
        )
        kwargs["weight"] = data["weight"]
        return cls.from_parameters(priorparameters, **kwargs)


class GaussianPrior(Prior):
    r"""One-dimensional Gaussian Prior.


    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution
        Default is 0
    sigma : float
        Standard deviation of the Gaussian distribution.
        Default is 1.
    """

    tag = ["GaussianPrior"]
    _type = "prior"
    mu = PriorParameter(name="mu", value=0)
    sigma = PriorParameter(name="sigma", value=1)

    @staticmethod
    def evaluate(value, mu, sigma):
        return ((value - mu) / sigma) ** 2


class UniformPrior(Prior):
    r"""Uniform Prior.

    Returns 1. if the parameter value is in (min, max).
    0. if otherwise.

    Parameters
    ----------
    min : float
        Minimum value.
        Default is -inf.

    max : float
        Maxmimum value.
        Default is inf.

    """

    tag = ["UniformPrior"]
    _type = "prior"
    min = PriorParameter(name="min", value=-np.inf, unit="")
    max = PriorParameter(name="max", value=np.inf, unit="")

    @staticmethod
    def evaluate(value, min, max):
        if min < value < max:
            return 1.0
        else:
            return 0.0

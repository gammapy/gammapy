# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Priors for Gammapy."""

import logging
import numpy as np
import astropy.units as u
from gammapy.modeling import Parameter, Parameters, PriorParameter, PriorParameters
from .core import ModelBase

__all__ = ["MultiVariateGaussianPrior", "GaussianPrior", "UniformPrior"]

log = logging.getLogger(__name__)


def _build_priorparameters_from_dict(data, default_parameters):
    """Build a `~gammapy.modeling.PriorParameters` object from input dictionary and default prior parameter values."""
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
    """Prior base class."""

    _unit = ""

    def __init__(self, modelparameters, **kwargs):
        if isinstance(modelparameters, Parameter):
            self._modelparameters = Parameters([modelparameters])
        elif isinstance(modelparameters, Parameters):
            self._modelparameters = modelparameters
        else:
            raise ValueError(f"Invalid model type {modelparameters}")

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

        for par in self._modelparameters:
            par.prior = self

    @property
    def modelparameters(self):
        return self._modelparameters

    @property
    def parameters(self):
        """Prior parameters as a `~gammapy.modeling.PriorParameters` object."""
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
        """Weight multiplied to the prior when evaluated."""
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    def __call__(self):
        """Call evaluate method."""
        # assuming the same unit as the PriorParameter here
        kwargs = {par.name: par.value for par in self.parameters}
        return self.weight * self.evaluate(self._modelparameters.value, **kwargs)

    def to_dict(self, full_output=False):
        """Create dictionary for YAML serialisation."""
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

        data = {
            "type": tag,
            "parameters": params,
            "weight": self.weight,
            "modelparameters": self._modelparameters,
        }

        return data

    @classmethod
    def from_dict(cls, data):
        """Get prior parameters from dictionary."""
        from . import PRIOR_REGISTRY

        prior_cls = PRIOR_REGISTRY.get_cls(data["type"])
        kwargs = {}

        if data["type"] not in prior_cls.tag:
            raise ValueError(
                f"Invalid model type {data['type']} for class {cls.__name__}"
            )
        priorparameters = _build_priorparameters_from_dict(
            data["parameters"], prior_cls.default_parameters
        )
        kwargs["weight"] = data["weight"]
        kwargs["modelparameters"] = data["modelparameters"]

        return prior_cls.from_parameters(priorparameters, **kwargs)


class MultiVariateGaussianPrior(Prior):
    """Multi-dimensional Gaussian Prior.

    Parameters
    ----------
    modelparameters :
        Meaning
    covariance_matrix :
        Meaning
    """

    tag = ["MultiVariateGaussianPrior"]
    _type = "prior"

    def __init__(self, modelparameters, covariance_matrix):
        # Why is this not being imported from the super??
        # self._modelparameters = modelparameters
        if isinstance(modelparameters, Parameter):
            self._modelparameters = Parameters([modelparameters])
        elif isinstance(modelparameters, Parameters):
            self._modelparameters = modelparameters
        else:
            raise ValueError(f"Invalid model type {modelparameters}")

        self._covariance_matrix = covariance_matrix

        # Ensure the correct shape
        value = np.asanyarray(self.covariance_matrix)
        npars = len(self._modelparameters)
        shape = (npars, npars)
        if value.shape != shape:
            raise ValueError(
                f"Invalid covariance matrix shape: {value.shape}, expected {shape}"
            )

        # Do we need this?
        self.dimension = value.shape[-1]

        for par in self._modelparameters:
            par.prior = self

        super().__init__(self._modelparameters)

    # def from_subcovariance(self, parameters, subcovar):
    #    idx = [parameters.index(par) for par in parameters]

    def __call__(self):
        """Call evaluate method"""
        return self.evaluate(self._modelparameters.value)

    @property
    def covariance_matrix(self):
        return self._covariance_matrix

    def evaluate(self, values):
        """Evaluate the MultiVariateGaussianPrior."""
        # Correct way to calculate that?
        return np.matmul(values, np.matmul(values, self.covariance_matrix))


class GaussianPrior(Prior):
    """One-dimensional Gaussian Prior.

    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution.
        Default is 0.
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
        """Evaluate the Gaussian prior."""
        return ((value - mu) / sigma) ** 2


class UniformPrior(Prior):
    """Uniform Prior.

    Returns 1 if the parameter value is in (min, max).
    0, if otherwise.

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
        """Evaluate the uniform prior."""
        if min < value < max:
            return 1.0
        else:
            return 0.0

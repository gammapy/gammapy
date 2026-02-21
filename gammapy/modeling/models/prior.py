# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Priors for Gammapy."""

import logging
import numpy as np
import astropy.units as u
from scipy.stats import norm, uniform, loguniform, gennorm
from gammapy.modeling import PriorParameter, PriorParameters
from .core import ModelBase

__all__ = [
    "GaussianPrior",
    "GeneralizedGaussianPrior",
    "UniformPrior",
    "LogUniformPrior",
    "Prior",
]

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
    """Prior abstract base class. For now, see existing examples of type of priors:

    - `GaussianPrior`
    - `UniformPrior`
    - `LogUniformPrior`
    """

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
        """Prior parameters as a `~gammapy.modeling.PriorParameters` object."""
        return PriorParameters(
            [getattr(self, name) for name in self.default_parameters.names]
        )

    def __init_subclass__(cls, **kwargs):
        # Add priorparameters list on the model sub-class (not instances)
        cls.default_parameters = PriorParameters(
            [_ for _ in cls.__dict__.values() if isinstance(_, PriorParameter)]
        )

    def _inverse_cdf(self, value):
        """Return inverse CDF for prior."""
        return self._random_variable.ppf(value)

    @property
    def weight(self):
        """Weight multiplied to the prior when evaluated."""
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    def __call__(self, value):
        """Call evaluate method."""
        # assuming the same unit as the PriorParameter here
        kwargs = {par.name: par.value for par in self.parameters}
        return self.weight * self.evaluate(value.value, **kwargs)

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

        data = {"type": tag, "parameters": params, "weight": self.weight}

        if self.type is None:
            return data
        else:
            return {self.type: data}

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Get prior parameters from dictionary."""
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
        init_kwargs = dict(kwargs)
        init_kwargs["weight"] = data["weight"]
        return cls.from_parameters(priorparameters, **init_kwargs)


class GaussianPrior(Prior):
    """One-dimensional Gaussian Prior.

    Parameters
    ----------
    mu : float, optional
        Mean of the Gaussian distribution.
        Default is 0.
    sigma : float, optional
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
        rv = norm(mu, sigma)
        return -2 * rv.logpdf(value)

    @property
    def _random_variable(self):
        """Return random variable object for prior."""
        return norm(self.mu.value, self.sigma.value)


class UniformPrior(Prior):
    """Uniform Prior.

    Returns 2log(max-min) if the parameter value is in [min, max].
    Returns inf, otherwise.
    Only well defined for finite values of min and max.

    Parameters
    ----------
    min : float, optional
        Minimum value.
        Default is 0.
    max : float, optional
        Maximum value.
        Default is 1.
    """

    tag = ["UniformPrior"]
    _type = "prior"
    min = PriorParameter(name="min", value=0.0, unit="")
    max = PriorParameter(name="max", value=1.0, unit="")

    @staticmethod
    def evaluate(value, min, max):
        """Evaluate the uniform prior."""
        rv = uniform(min, max - min)
        return -2 * rv.logpdf(value)

    @property
    def _random_variable(self):
        """Return random variable object for prior."""
        return uniform(self.min.value, self.max.value - self.min.value)


class LogUniformPrior(Prior):
    """LogUniform Prior.

    Equivalent to a uniform prior on the log of the parameter

    Parameters
    ----------
    min : float, optional
        Minimum value. Default is 1e-14.
    max : float, optional
        Maximum value. Default is 1e-10.
    """

    tag = ["LogUniformPrior"]
    _type = "prior"
    min = PriorParameter(name="min", value=1e-14, unit="")
    max = PriorParameter(name="max", value=1e-10, unit="")

    @staticmethod
    def evaluate(value, min, max):
        """
        Evaluate the likelihood penalization term (hence -2*).
        Note that this is currently a different scaling that the Uniform or Gaussian priors.
        With current implementation the TS of a source with/without LogUniform prior would be different... TBD
        """
        rv = loguniform(min, max)
        return -2 * rv.logpdf(value)

    @property
    def _random_variable(self):
        """Return random variable object for prior."""
        return loguniform(self.min.value, self.max.value)


class GeneralizedGaussianPrior(Prior):
    """One-dimensional Generalized Gaussian Prior.

    Parameters
    ----------
    mu : float, optional
        Mean of the Gaussian distribution.
        Default is 0.
    sigma : float, optional
        Standard deviation of the Gaussian distribution.
        Default is 1.
    eta : `float`, optional
        eta is a shape parameter
        For eta=1 it is identical to a Laplace distribution (scaled by sqrt(2)).
        For eta=0.5 it is identical to a normal distribution.
        Default is 0.5.
    """

    tag = ["GeneralizedGaussianPrior"]
    _type = "prior"
    mu = PriorParameter(name="mu", value=0)
    sigma = PriorParameter(name="sigma", value=1)
    eta = PriorParameter(name="eta", value=0.5)

    @staticmethod
    def evaluate(value, mu, sigma, eta):
        """Evaluate the Gaussian prior."""
        rv = gennorm(beta=1.0 / eta, loc=mu, scale=sigma * np.sqrt(2))
        return -2 * rv.logpdf(value)

    @property
    def _random_variable(self):
        """Return random variable object for prior."""
        return gennorm(
            beta=1.0 / self.eta.value,
            loc=self.mu.value,
            scale=self.sigma.value * np.sqrt(2),
        )

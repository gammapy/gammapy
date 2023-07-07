import numpy as np
import astropy.units as u
from gammapy.modeling.models import ModelBase
from gammapy.utils.table import table_from_row_data
from . import Parameter, Parameters


class PriorParameter(Parameter):
    def __init__(
        self,
        name,
        value,
        unit="",
        scale=1,
        min=np.nan,
        max=np.nan,
    ):
        if not isinstance(name, str):
            raise TypeError(f"Name must be string, got '{type(name)}' instead")

        self._name = name
        self.scale = scale
        self.min = min
        self.max = max
        self._error = np.nan
        if isinstance(value, u.Quantity) or isinstance(value, str):
            val = u.Quantity(value)
            self.value = val.value
            self.unit = val.unit
        else:
            self.factor = value
            self.unit = unit
        self._type = "prior"

    def to_dict(self):
        """Convert to dict."""
        output = {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.to_string("fits"),
            "min": self.min,
            "max": self.max,
        }
        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name!r}, value={self.value!r}, "
            f"factor={self.factor!r}, scale={self.scale!r}, unit={self.unit!r}, "
            f"min={self.min!r}, max={self.max!r})"
        )


class PriorParameters(Parameters):
    def __init__(self, parameters=None):
        if parameters is None:
            parameters = []
        else:
            parameters = list(parameters)

        self._parameters = parameters

    def to_table(self):
        """Convert parameter attributes to `~astropy.table.Table`."""
        rows = []
        for p in self._parameters:
            d = p.to_dict()
            rows.append({**dict(type=p.type), **d})
        table = table_from_row_data(rows)

        table["value"].format = ".4e"
        for name in ["min", "max"]:
            table[name].format = ".3e"

        return table


class PriorModel(ModelBase):
    """
    Base prior class.
    """

    _weight = 1
_type = "prior"

    @property
    def parameters(self):
        """PriorParameters (`~gammapy.modeling.PriorParameters`)"""
        return PriorParameters(
            [getattr(self, name) for name in self.default_parameters.names]
        )

    def __init_subclass__(cls, **kwargs):
        # Add parameters list on the model sub-class (not instances)
        cls.default_parameters = PriorParameters(
            [_ for _ in cls.__dict__.values() if isinstance(_, PriorParameter)]
        )

    # for now only one unit which has to be set
    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value
        for p in self.parameters:
            p.unit = self._unit

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    def __call__(self, value):
        """Call evaluate method"""
        kwargs = {par.name: par.quantity.to(self.unit) for par in self.parameters}
        if isinstance(value, Parameter):
            return self.evaluate(value.quantity.to(self.unit), **kwargs)
        elif isinstance(value, Parameters):
            return self.evaluate(
                [v.quantity.to(self.unit).value for v in value], **kwargs
            )
        else:
            raise TypeError(f"Invalid type: {value}, {type(value)}")

    def __str__(self):
        string = f"{self.__class__.__name__}\n"
        string += f"unit: {self.unit}\n"
        string += f"weight: {self.weight}\n"

        if len(self.parameters) > 0:
            string += f"\n{self.parameters.to_table()}"
        return string

    def __repr__(self):
        return self.__class__.__name__


class GaussianPrior(PriorModel):
    """Gaussian Prior with mu and sigma."""

    tag = ["GaussianPrior"]
    mu = PriorParameter(name="mu", value=0, unit="")
    sigma = PriorParameter(name="sigma", value=1, unit="")

    @staticmethod
    def evaluate(value, mu, sigma):
        return ((value - mu) / sigma) ** 2


class UniformPrior(PriorModel):
    """Uniform Prior"""

    tag = ["UniformPrior"]
    uni = PriorParameter(name="uni", value=0, min=0, max=10, unit="")

    @staticmethod
    def evaluate(value, uni):
        return uni


class CovarianceGaussianPrior(PriorModel):
    """Gaussian Priors on mulitple parameters with different mu and sigma. Set on a model.

    Parameters
    ----------
    priorparameters : PriorParameters (`~gammapy.modeling.PriorParameters`)
        Parameters with the covariance matrix set.
    """

    tag = ["CovarianceGaussianPrior"]

    def __init__(self, priorparameters):
        self.default_parameters = priorparameters
        super().__init__()

    @property
    def cov(self):
        return self.default_parameters.covariance

    def evaluate(self, values, **pars):
        from numpy import linalg

        return values @ linalg.inv(self.cov.data) @ values

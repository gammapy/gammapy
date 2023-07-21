from gammapy.modeling import PriorParameter, PriorParameters
from .core import ModelBase


class PriorModel(ModelBase):
    @property
    def parameters(self):
        """Parameters (`~gammapy.modeling.Parameters`)"""
        return PriorParameters(
            [getattr(self, name) for name in self.default_parameters.names]
        )

    def __init_subclass__(cls, **kwargs):
        # Add parameters list on the model sub-class (not instances)
        cls.default_parameters = PriorParameters(
            [_ for _ in cls.__dict__.values() if isinstance(_, PriorParameter)]
        )

    # for now only one unit
    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value
        for p in self.parameters:
            p.unit = self._unit

    def __call__(self, value):
        """Call evaluate method"""
        kwargs = {par.name: par.quantity.to(self.unit) for par in self.parameters}
        return self.evaluate(value, **kwargs)

    def to_dict(self):
        return dict()


class GaussianPrior(PriorModel):
    """Gaussian Prior with mu and sigma."""

    tag = ["GaussianPrior"]
    _type = "prior"
    mu = PriorParameter(name="mu", value=0)
    sigma = PriorParameter(name="sigma", value=1)

    @staticmethod
    def evaluate(value, mu, sigma):
        return ((value.quantity.to(mu.unit) - mu) / sigma) ** 2


class UniformPrior(PriorModel):
    """Uniform Prior"""

    tag = ["UniformPrior"]
    uni = PriorParameter(name="uni", value=0, min=0, max=10, unit="")

    @staticmethod
    def evaluate(value, uni):
        return uni

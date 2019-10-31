# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import astropy.units as u
from .parameter import Parameter, Parameters

__all__ = ["Model"]


class Model:
    """Model base class."""

    def __init__(self, **kwargs):
        # Copy default parameters from the class to the instance
        self._parameters = self.__class__.default_parameters.copy()
        for parameter in self._parameters:
            if parameter.name in self.__dict__:
                raise ValueError(
                    f"Invalid parameter name: {parameter.name!r}."
                    f"Attribute exists already: {getattr(self, parameter.name)!r}"
                )

            setattr(self, parameter.name, parameter)

        # Update parameter information from kwargs
        for name, value in kwargs.items():
            if name not in self.parameters.names:
                raise ValueError(
                    f"Invalid argument: {name!r}. Parameter names are: {self.parameters.names}"
                )

            self._parameters[name].quantity = u.Quantity(value)

    def __init_subclass__(cls, **kwargs):
        # Add parameters list on the model sub-class (not instances)
        cls.default_parameters = Parameters(
            [_ for _ in cls.__dict__.values() if isinstance(_, Parameter)]
        )

    def _init_from_parameters(self, parameters):
        """Create model from list of parameters.

        This should be called for models that generate
        the parameters dynamically in ``__init__``,
        like the ``NaimaSpectralModel``
        """
        # TODO: should we pass through `Parameters` here? Why?
        parameters = Parameters(parameters)
        self._parameters = parameters
        for parameter in parameters:
            setattr(self, parameter.name, parameter)

    @property
    def parameters(self):
        """Parameters (`~gammapy.modeling.Parameters`)"""
        return self._parameters

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    def __str__(self):
        return f"{self.__class__.__name__}\n\n{self.parameters.to_table()}"

    def to_dict(self):
        """Create dict for YAML serialisation"""
        return {"type": self.tag, "parameters": self.parameters.to_dict()["parameters"]}

    @classmethod
    def from_dict(cls, data):
        params = {
            x["name"].split("@")[0]: x["value"] * u.Unit(x["unit"])
            for x in data["parameters"]
        }

        # TODO: this is a special case for spatial models, maybe better move to `SpatialModel` base class
        if "frame" in data:
            params["frame"] = data["frame"]

        model = cls(**params)
        model._update_from_dict(data)
        return model

    # TODO: try to get rid of this
    def _update_from_dict(self, data):
        self._parameters = Parameters.from_dict(data)
        for parameter in self.parameters:
            setattr(self, parameter.name, parameter)

    @staticmethod
    def create(tag, *args, **kwargs):
        """Create a model instance.

        Examples
        --------
        >>> from gammapy.modeling import Model
        >>> spectral_model = Model.create("PowerLaw2SpectralModel", amplitude="1e-10 cm-2 s-1", index=3)
        >>> type(spectral_model)
        gammapy.modeling.models.spectral.PowerLaw2SpectralModel
        """
        from .models import MODELS

        cls = MODELS.get_cls(tag)
        return cls(*args, **kwargs)

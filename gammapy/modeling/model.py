# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import astropy.units as u
from .parameter import Parameters

__all__ = ["Model"]


class Model:
    """Model base class."""

    __slots__ = ["_parameters"]

    def __init__(self, params=None):
        self._parameters = Parameters(params)

    @property
    def parameters(self):
        """Parameters (`~gammapy.modeling.Parameters`)"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    def __str__(self):
        return f"{self.__class__.__name__}\n\n" f"{self.parameters.to_table()}"

    def to_dict(self):
        """Create dict for YAML serilisation"""
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

    def _update_from_dict(self, data):
        self.parameters = Parameters.from_dict(data)
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

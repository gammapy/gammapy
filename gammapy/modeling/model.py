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

    def to_dict(self):
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

        init = cls(**params)
        init.parameters = Parameters.from_dict(data)
        for parameter in init.parameters.parameters:
            setattr(init, parameter.name, parameter)
        return init

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

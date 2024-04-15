# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""astromodels wrapper."""

import numpy as np
import astropy.units as u
from gammapy.modeling import Parameter, Parameters, PriorParameter, PriorParameters
from gammapy.modeling.models.prior import Prior
from gammapy.modeling.models.spatial import SpatialModel
from gammapy.modeling.models.spectral import SpectralModel


class AstroModelMixin:

    def _init_instance(self, module, function, **kwargs):
        self._func_name = function
        self._func = getattr(module, function)
        self._instance = self._func(**kwargs)

    def _set_default_parameters(self):
        parameters_list = []
        for key, param_data in self._instance.parameters.items():
            if self.type == "prior":
                parameter = PriorParameter(
                    name=key, value=param_data.value, unit=param_data.unit
                )
            else:
                parameter = Parameter(
                    name=key,
                    value=param_data.value,
                    unit=param_data.unit,
                    frozen=not param_data.free,
                )
            if param_data.min_value:
                parameter.min = param_data.min_value
            if param_data.max_value:
                parameter.max = param_data.max_value
            parameters_list.append(parameter)

        if self.type == "prior":
            self.default_parameters = PriorParameters(parameters_list)
        else:
            self.default_parameters = Parameters(parameters_list)

    def to_dict(self, full_output=True):
        data = super().to_dict(full_output=full_output)
        data[self.type]["function"] = self._func_name
        return data

    @classmethod
    def from_dict(cls, data, **kwargs):
        data = data[cls._type]
        parameters = Parameters.from_dict(data["parameters"])
        kwargs = {p.name: p.value for p in parameters}
        if "frame" in data:
            model = cls(data["function"], data["frame"], **kwargs)
        else:
            model = cls(data["function"], **kwargs)
        model.default_parameters = parameters
        return model


class AstroSpectralModel(AstroModelMixin, SpectralModel):

    tag = ["AstroSpectralModel"]

    def __init__(self, function, **kwargs):
        import astromodels.functions.functions_1D as module

        self._init_instance(module, function, **kwargs)
        self._instance._x_unit = u.Unit("TeV")
        self._instance._y_unit = u.Unit("TeV-1 s-1 cm-2")
        self._set_default_parameters()
        super().__init__()

    def evaluate(self, energy, **kwargs):
        energy = u.Quantity(np.atleast_1d(energy))
        args = [val * u.Unit("") for val in kwargs.values()]
        return self._instance.evaluate(energy, *args)


class AstroSpatialModel(AstroModelMixin, SpatialModel):

    tag = ["AstroSpatialModel"]

    def __init__(self, function, frame="icrs", **kwargs):
        import astromodels.functions.functions_2D as module

        self.is_energy_dependent = not hasattr(module, function)
        if self.is_energy_dependent:
            import astromodels.functions.functions_3D as module
        self._init_instance(module, function, **kwargs)
        if self.is_energy_dependent:
            self._instance.set_units(u.deg, u.deg, u.TeV, None)
        else:
            self._instance.set_units(u.deg, u.deg, None)
        self._instance._frame = frame
        self._set_default_parameters()
        self.frame = frame

        super().__init__()

    def evaluate(self, lon, lat, energy=None, **kwargs):
        lon = u.Quantity(np.atleast_1d(lon))
        lat = u.Quantity(np.atleast_1d(lat))
        args = [val * u.Unit("") for val in kwargs.values()]
        if self.is_energy_dependent:
            energy = u.Quantity(np.atleast_1d(energy))
            return self._instance.evaluate(lon, lat, energy, *args)
        else:
            return self._instance.evaluate(lon, lat, *args).to("sr-1")

    @property
    def lon_0(self):
        return self.lon0

    @property
    def lat_0(self):
        return self.lat0


class AstroPriorModel(AstroModelMixin, Prior):

    tag = ["AstroPriorModel"]
    _type = "prior"

    def __init__(self, function, **kwargs):
        import astromodels.functions.priors as module

        self._init_instance(module, function, **kwargs)
        self._set_default_parameters()
        super().__init__()

    def evaluate(self, value, **kwargs):
        value = u.Quantity(np.atleast_1d(value))
        args = [val * u.Unit("") for val in kwargs.values()]
        return self._instance.evaluate(value, *args)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import collections.abc
import copy
from pathlib import Path
import astropy.units as u
import numpy as np
import yaml
from gammapy.modeling import Parameter, Parameters, Covariance
from gammapy.utils.scripts import make_path, make_name


def _set_link(shared_register, model):
    for param in model.parameters:
        name = param.name
        link_label = param._link_label_io
        if link_label is not None:
            if link_label in shared_register:
                new_param = shared_register[link_label]
                model.parameters.link(name, new_param)
            else:
                shared_register[link_label] = param
    return shared_register


__all__ = ["Model", "Models"]


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

        self._covariance = Covariance(self.parameters)

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
    def covariance(self):
        for par in self._parameters:
            pars = Parameters([par])
            covar = Covariance(pars, data=[[par.error ** 2]])
            self._covariance.set_subcovariance(covar)

        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._covariance.data = covariance

        for par in self.parameters:
            pars = Parameters([par])
            variance = self._covariance.get_subcovariance(pars)
            par.error = np.sqrt(variance)

    @property
    def parameters(self):
        """Parameters (`~gammapy.modeling.Parameters`)"""
        # trigger recursive update
        return self._parameters

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    def to_dict(self):
        """Create dict for YAML serialisation"""
        return {"type": self.tag, "parameters": self.parameters.to_dict()["parameters"]}

    @classmethod
    def from_dict(cls, data):
        params = {x["name"]: x["value"] * u.Unit(x["unit"]) for x in data["parameters"]}

        # TODO: this is a special case for spatial models, maybe better move to `SpatialModel` base class
        if "frame" in data:
            params["frame"] = data["frame"]

        model = cls(**params)
        model._update_from_dict(data)
        return model

    # TODO: try to get rid of this
    def _update_from_dict(self, data):
        self._parameters.update_from_dict(data)
        for parameter in self.parameters:
            setattr(self, parameter.name, parameter)

    @staticmethod
    def create(tag, *args, **kwargs):
        """Create a model instance.

        Examples
        --------
        >>> from gammapy.modeling.models import Model
        >>> spectral_model = Model.create("PowerLaw2SpectralModel", amplitude="1e-10 cm-2 s-1", index=3)
        >>> type(spectral_model)
        gammapy.modeling.models.spectral.PowerLaw2SpectralModel
        """
        from . import MODELS

        cls = MODELS.get_cls(tag)
        return cls(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}\n\n{self.parameters.to_table()}"


class Models(collections.abc.MutableSequence):
    """Sky model collection.

    Parameters
    ----------
    models : `SkyModel`, list of `SkyModel` or `Models`
        Sky models
    """

    def __init__(self, models=None):
        if models is None:
            models = []

        if isinstance(models, Models):
            models = models._models
        elif isinstance(models, Model):
            models = [models]
        elif isinstance(models, list):
            models = models
        else:
            raise TypeError(f"Invalid type: {models!r}")

        unique_names = []
        for model in models:
            if model.name in unique_names:
                raise (ValueError("Model names must be unique"))
            unique_names.append(model.name)

        self._models = models
        self._covariance = Covariance(self.parameters)

    def _check_covariance(self):
        if not self.parameters == self._covariance.parameters:
            self._covariance = Covariance.from_stack(
                [model.covariance for model in self._models]
            )

    @property
    def covariance(self):
        self._check_covariance()

        for model in self._models:
            self._covariance.set_subcovariance(model.covariance)

        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._check_covariance()
        self._covariance.data = covariance

        for model in self._models:
            subcovar = self._covariance.get_subcovariance(
                model.covariance.parameters
            )
            model.covariance = subcovar

    @property
    def parameters(self):
        return Parameters.from_stack([_.parameters for _ in self._models])

    @property
    def names(self):
        return [m.name for m in self._models]

    @classmethod
    def read(cls, filename):
        """Read from YAML file."""
        yaml_str = Path(filename).read_text()
        return cls.from_yaml(yaml_str)

    @classmethod
    def from_yaml(cls, yaml_str):
        """Create from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        """Create from dict."""
        from . import MODELS, SkyModel

        models = []
        for component in data["components"]:
            model = MODELS.get_cls(component["type"]).from_dict(component)
            models.append(model)
        models = cls(models)

        shared_register = {}
        for model in models:
            if isinstance(model, SkyModel):
                submodels = [
                    model.spectral_model,
                    model.spatial_model,
                    model.temporal_model,
                ]
                for submodel in submodels:
                    if submodel is not None:
                        shared_register = _set_link(shared_register, submodel)
            else:
                shared_register = _set_link(shared_register, model)
        return models

    def write(self, path, overwrite=False):
        """Write to YAML file."""
        path = make_path(path)
        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")
        path.write_text(self.to_yaml())

    def to_yaml(self):
        """Convert to YAML string."""
        data = self.to_dict()
        return yaml.dump(
            data, sort_keys=False, indent=4, width=80, default_flow_style=None
        )

    def to_dict(self):
        """Convert to dict."""
        # update linked parameters labels
        params_list = []
        params_shared = []
        for param in self.parameters:
            if param not in params_list:
                params_list.append(param)
            elif param not in params_shared:
                params_shared.append(param)
        for param in params_shared:
            param._link_label_io = param.name + "@" + make_name()

        models_data = []
        for model in self._models:
            model_data = model.to_dict()
            models_data.append(model_data)
        return {"components": models_data}

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n\n"

        for idx, model in enumerate(self):
            str_ += f"Component {idx}: "
            str_ += str(model)

        return str_.expandtabs(tabsize=2)

    def __add__(self, other):
        if isinstance(other, (Models, list)):
            return Models([*self, *other])
        elif isinstance(other, Model):
            if other.name in self.names:
                raise (ValueError("Model names must be unique"))
            return Models([*self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def __getitem__(self, key):
        return self._models[self.index(key)]

    def __delitem__(self, key):
        del self._models[self.index(key)]

    def __setitem__(self, key, model):
        from gammapy.modeling.models import SkyModel, SkyDiffuseCube

        if isinstance(model, (SkyModel, SkyDiffuseCube)):
            self._models[self.index(key)] = model
        else:
            raise TypeError(f"Invalid type: {model!r}")

    def insert(self, idx, model):
        if model.name in self.names:
            raise (ValueError("Model names must be unique"))

        self._models.insert(idx, model)

    def index(self, key):
        if isinstance(key, (int, slice)):
            return key
        elif isinstance(key, str):
            return self.names.index(key)
        elif isinstance(key, Model):
            return self._models.index(key)
        else:
            raise TypeError(f"Invalid type: {type(key)!r}")

    def __len__(self):
        return len(self._models)

    def _ipython_key_completions_(self):
        return self.names

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)
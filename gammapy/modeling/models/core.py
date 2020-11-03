# Licensed under a 3-clause BSD style license - see LICENSE.rst
import collections.abc
import copy
from os.path import split
import numpy as np
import astropy.units as u
from astropy.table import Table
import yaml
from gammapy.modeling import Covariance, Parameter, Parameters
from gammapy.utils.scripts import make_name, make_path


def _set_link(shared_register, model):
    for param in model.parameters:
        name = param.name
        link_label = param._link_label_io
        if link_label is not None:
            if link_label in shared_register:
                new_param = shared_register[link_label]
                setattr(model, name, new_param)
            else:
                shared_register[link_label] = param
    return shared_register


__all__ = ["Model", "Models", "DatasetModels"]


class Model:
    """Model base class."""

    def __init__(self, **kwargs):
        # Copy default parameters from the class to the instance
        default_parameters = self.default_parameters.copy()

        for par in default_parameters:
            value = kwargs.get(par.name, par)

            if not isinstance(value, Parameter):
                par.quantity = u.Quantity(value)
            else:
                par = value

            setattr(self, par.name, par)

        self._covariance = Covariance(self.parameters)

    def __init_subclass__(cls, **kwargs):
        # Add parameters list on the model sub-class (not instances)
        cls.default_parameters = Parameters(
            [_ for _ in cls.__dict__.values() if isinstance(_, Parameter)]
        )

    @classmethod
    def from_parameters(cls, parameters, **kwargs):
        """Create model from parameter list

        Parameters
        ----------
        parameters : `Parameters`
            Parameters for init

        Returns
        -------
        model : `Model`
            Model instance
        """
        for par in parameters:
            kwargs[par.name] = par
        return cls(**kwargs)

    def _check_covariance(self):
        if not self.parameters == self._covariance.parameters:
            self._covariance = Covariance(self.parameters)

    @property
    def covariance(self):
        self._check_covariance()
        for par in self.parameters:
            pars = Parameters([par])
            error = np.nan_to_num(par.error ** 2, nan=1)
            covar = Covariance(pars, data=[[error]])
            self._covariance.set_subcovariance(covar)

        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._check_covariance()
        self._covariance.data = covariance

        for par in self.parameters:
            pars = Parameters([par])
            variance = self._covariance.get_subcovariance(pars)
            par.error = np.sqrt(variance)

    @property
    def parameters(self):
        """Parameters (`~gammapy.modeling.Parameters`)"""
        return Parameters(
            [getattr(self, name) for name in self.default_parameters.names]
        )

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    def to_dict(self, full_output=False):
        """Create dict for YAML serialisation"""
        tag = self.tag[0] if isinstance(self.tag, list) else self.tag
        params = self.parameters.to_dict()

        if not full_output:
            for par, par_default in zip(params, self.default_parameters):
                init = par_default.to_dict()
                for item in ["min", "max", "error"]:
                    default = init[item]

                    if par[item] == default or np.isnan(default):
                        del par[item]

                if not par["frozen"]:
                    del par["frozen"]

                if init["unit"] == "":
                    del par["unit"]

        return {"type": tag, "parameters": params}

    @classmethod
    def from_dict(cls, data):
        kwargs = {}

        par_data = []

        for par, par_yaml in zip(cls.default_parameters, data["parameters"]):
            par_dict = par.to_dict()
            par_dict.update(par_yaml)
            par_data.append(par_dict)

        parameters = Parameters.from_dict(par_data)

        # TODO: this is a special case for spatial models, maybe better move to `SpatialModel` base class
        if "frame" in data:
            kwargs["frame"] = data["frame"]

        return cls.from_parameters(parameters, **kwargs)

    @staticmethod
    def create(tag, model_type=None, *args, **kwargs):
        """Create a model instance.

        Examples
        --------
        >>> from gammapy.modeling.models import Model
        >>> spectral_model = Model.create("pl-2", model_type="spectral", amplitude="1e-10 cm-2 s-1", index=3)
        >>> type(spectral_model)
        gammapy.modeling.models.spectral.PowerLaw2SpectralModel
        """
        from . import (
            MODEL_REGISTRY,
            SPATIAL_MODEL_REGISTRY,
            SPECTRAL_MODEL_REGISTRY,
            TEMPORAL_MODEL_REGISTRY,
        )

        if model_type is None:
            cls = MODEL_REGISTRY.get_cls(tag)
        else:
            registry = {
                "spatial": SPATIAL_MODEL_REGISTRY,
                "spectral": SPECTRAL_MODEL_REGISTRY,
                "temporal": TEMPORAL_MODEL_REGISTRY,
            }
            cls = registry[model_type].get_cls(tag)
        return cls(*args, **kwargs)

    def __str__(self):
        string = f"{self.__class__.__name__}\n"
        if len(self.parameters) > 0:
            string += f"\n{self.parameters.to_table()}"
        return string


class DatasetModels(collections.abc.Sequence):
    """Immutable models container

    Parameters
    ----------
    models : `SkyModel`, list of `SkyModel` or `Models`
        Sky models
    """

    def __init__(self, models=None):
        if models is None:
            models = []

        if isinstance(models, (Models, DatasetModels)):
            models = models._models
        elif isinstance(models, Model):
            models = [models]
        elif not isinstance(models, list):
            raise TypeError(f"Invalid type: {models!r}")

        unique_names = []
        for model in models:
            if model.name in unique_names:
                raise (ValueError("Model names must be unique"))
            unique_names.append(model.name)

        self._models = models
        self._covar_file = None
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
            subcovar = self._covariance.get_subcovariance(model.covariance.parameters)
            model.covariance = subcovar

    @property
    def parameters(self):
        return Parameters.from_stack([_.parameters for _ in self._models])

    @property
    def parameters_unique_names(self):
        """List of unique parameter names as model_name.par_type.par_name"""
        names = []
        for model in self:
            for par in model.parameters:
                components = [model.name, par.type, par.name]
                name = ".".join(components)
                names.append(name)

        return names

    @property
    def names(self):
        return [m.name for m in self._models]

    @classmethod
    def read(cls, filename):
        """Read from YAML file."""
        yaml_str = make_path(filename).read_text()
        path, filename = split(filename)
        return cls.from_yaml(yaml_str, path=path)

    @classmethod
    def from_yaml(cls, yaml_str, path=""):
        """Create from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data, path=path)

    @classmethod
    def from_dict(cls, data, path=""):
        """Create from dict."""
        from . import MODEL_REGISTRY, SkyModel

        models = []

        for component in data["components"]:
            model_cls = MODEL_REGISTRY.get_cls(component["type"])
            model = model_cls.from_dict(component)
            models.append(model)

        models = cls(models)

        if "covariance" in data:
            filename = data["covariance"]
            path = make_path(path)
            if not (path / filename).exists():
                path, filename = split(filename)

            models.read_covariance(path, filename, format="ascii.fixed_width")

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

    def write(self, path, overwrite=False, full_output=False, write_covariance=True):
        """Write to YAML file.

        Parameters
        ----------
        path : `pathlib.Path` or str
            path to write files
        overwrite : bool
            overwrite files
        write_covariance : bool
            save covariance or not
        """
        base_path, _ = split(path)
        path = make_path(path)
        base_path = make_path(base_path)

        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")

        if (
            write_covariance
            and self.covariance is not None
            and len(self.parameters) != 0
        ):
            filecovar = path.stem + "_covariance.dat"
            kwargs = dict(
                format="ascii.fixed_width", delimiter="|", overwrite=overwrite
            )
            self.write_covariance(base_path / filecovar, **kwargs)
            self._covar_file = filecovar

        path.write_text(self.to_yaml(full_output))

    def to_yaml(self, full_output=False):
        """Convert to YAML string."""
        data = self.to_dict(full_output)
        return yaml.dump(
            data, sort_keys=False, indent=4, width=80, default_flow_style=False
        )

    def to_dict(self, full_output=False):
        """Convert to dict."""
        # update linked parameters labels
        params_list = []
        params_shared = []
        for param in self.parameters:
            if param not in params_list:
                params_list.append(param)
                params_list.append(param)
            elif param not in params_shared:
                params_shared.append(param)
        for param in params_shared:
            param._link_label_io = param.name + "@" + make_name()

        models_data = []
        for model in self._models:
            model_data = model.to_dict(full_output)
            models_data.append(model_data)
        if self._covar_file is not None:
            return {
                "components": models_data,
                "covariance": str(self._covar_file),
            }
        else:
            return {"components": models_data}

    def read_covariance(self, path, filename="_covariance.dat", **kwargs):
        """Read covariance data from file

        Parameters
        ----------
        filename : str
            Filename
        **kwargs : dict
            Keyword arguments passed to `~astropy.table.Table.read`

        """
        path = make_path(path)
        filepath = str(path / filename)
        t = Table.read(filepath, **kwargs)
        t.remove_column("Parameters")
        arr = np.array(t)
        data = arr.view(float).reshape(arr.shape + (-1,))
        self.covariance = data
        self._covar_file = filename

    def write_covariance(self, filename, **kwargs):
        """Write covariance to file

        Parameters
        ----------
        filename : str
            Filename
        **kwargs : dict
            Keyword arguments passed to `~astropy.table.Table.write`

        """
        names = self.parameters_unique_names
        table = Table()
        table["Parameters"] = names

        for idx, name in enumerate(names):
            values = self.covariance.data[idx]
            table[name] = values

        table.write(make_path(filename), **kwargs)

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

    def select(self, dataset_name=None, tag=None, name_substring=None):
        """Select subset of models correspondiog to a given dataset

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        tag : str
            Model tag
        name_substring : str
            Substring contained in the model name

        Returns
        -------
        dataset_model : `DatasetModels`
            Dataset models
        """
        models = []

        for model in self:
            selection = True

            if dataset_name:
                selection &= (
                    model.datasets_names is None or dataset_name in model.datasets_names
                )

            if tag:
                selection &= tag in model.tag

            if name_substring:
                selection &= name_substring in model.name

            if selection:
                models.append(model)

        return self.__class__(models)


class Models(DatasetModels, collections.abc.MutableSequence):
    """Sky model collection.

    Parameters
    ----------
    models : `SkyModel`, list of `SkyModel` or `Models`
        Sky models
    """

    def __delitem__(self, key):
        del self._models[self.index(key)]

    def __setitem__(self, key, model):
        from gammapy.modeling.models import SkyModel, FoVBackgroundModel

        if isinstance(model, (SkyModel, FoVBackgroundModel)):
            self._models[self.index(key)] = model
        else:
            raise TypeError(f"Invalid type: {model!r}")

    def insert(self, idx, model):
        if model.name in self.names:
            raise (ValueError("Model names must be unique"))

        self._models.insert(idx, model)

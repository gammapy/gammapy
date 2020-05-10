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


__all__ = ["Model", "Models", "ProperModels"]


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
            covar = Covariance(pars, data=[[par.error ** 2]])
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

    def to_dict(self):
        """Create dict for YAML serialisation"""
        return {"type": self.tag, "parameters": self.parameters.to_dict()}

    @classmethod
    def from_dict(cls, data):
        kwargs = {}
        parameters = Parameters.from_dict(data["parameters"])

        # TODO: this is a special case for spatial models, maybe better move to `SpatialModel` base class
        if "frame" in data:
            kwargs["frame"] = data["frame"]

        return cls.from_parameters(parameters, **kwargs)

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
        from gammapy.modeling.models import SkyModel

        param_names = []
        for m in self._models:
            if isinstance(m, SkyModel):
                for p in m.parameters:
                    if (
                        m.spectral_model is not None
                        and p in m.spectral_model.parameters
                    ):
                        tag = ".spectral."
                    elif (
                        m.spatial_model is not None and p in m.spatial_model.parameters
                    ):
                        tag = ".spatial."
                    elif (
                        m.temporal_model is not None
                        and p in m.temporal_model.parameters
                    ):
                        tag = ".temporal."
                    param_names.append(m.name + tag + p.name)
            else:
                for p in m.parameters:
                    param_names.append(m.name + "." + p.name)
        return param_names

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
        from . import MODELS, SkyModel

        models = []

        for component in data["components"]:
            model = MODELS.get_cls(component["type"]).from_dict(component)
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

    def write(self, path, overwrite=False):
        """Write to YAML file."""
        base_path, _ = split(path)
        path = make_path(path)
        base_path = make_path(base_path)

        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")

        if self.covariance is not None and len(self.parameters) != 0:
            filecovar = path.stem + "_covariance.dat"
            kwargs = dict(
                format="ascii.fixed_width", delimiter="|", overwrite=overwrite
            )
            self.write_covariance(base_path / filecovar, **kwargs)
            self._covar_file = filecovar

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


class ProperModels(Models):
    """ Proper Models of a Dataset or Datasets."""

    def __init__(self, parent):
        from gammapy.datasets import Dataset, Datasets

        if isinstance(parent, Dataset):
            self._datasets = [parent]
            self._is_dataset = True
        elif isinstance(parent, Datasets):
            self._datasets = parent._datasets
            self._is_dataset = False
        else:
            raise TypeError(f"Invalid type: {type(parent)!r}")

        unique_models = []
        for d in self._datasets:
            if d._models is not None:
                for model in d._models:
                    if model not in unique_models:
                        if (
                            model.datasets_names is None
                            or d.name in model.datasets_names
                        ):
                            unique_models.append(model)
            else:
                d._models = []
            self._models = unique_models

        self._covar_file = None
        self._covariance = Covariance(self.parameters)

    def __add__(self, other):
        if isinstance(other, (Models, list)):
            pass
        elif isinstance(other, Model):
            other = [other]
        else:
            raise TypeError(f"Invalid type: {other!r}")
        for d in self._datasets:
            for m in other:
                if m not in d._models:
                    d._models.append(m)
                if (
                    m.datasets_names is not None
                    and d.name not in m.datasets_names
                    and self._is_dataset
                ):
                    m.datasets_names.append(d.name)

    def __delitem__(self, key):
        for d in self._datasets:
            if key in d.models.names:
                datasets_names = d.models[key].datasets_names
                if datasets_names is None or d.name in datasets_names:
                    d._models.remove(key)

    def __setitem__(self, key, model):
        from gammapy.modeling.models import SkyModel, SkyDiffuseCube

        for d in self._datasets:
            if model in d._models:
                if isinstance(model, (SkyModel, SkyDiffuseCube)):
                    d._models[key] = model
                else:
                    raise TypeError(f"Invalid type: {model!r}")
            if (
                model.datasets_names is not None
                and d.name not in model.datasets_names
                and self._is_dataset
            ):
                model.datasets_names.append(d.name)

    def insert(self, idx, model):
        from gammapy.modeling.models import SkyModel, SkyDiffuseCube

        for d in self._datasets:
            if model.name not in d._models.names:
                if isinstance(model, (SkyModel, SkyDiffuseCube)):
                    if idx == len(self):
                        index = len(d._models)
                    else:
                        index = idx
                    d._models.insert(index, model)
                else:
                    raise TypeError(f"Invalid type: {model!r}")
            if (
                model.datasets_names is not None
                and d.name not in model.datasets_names
                and self._is_dataset
            ):
                model.datasets_names.append(d.name)

    def remove(self, value):
        key = value.name
        del self[key]

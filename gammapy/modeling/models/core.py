# Licensed under a 3-clause BSD style license - see LICENSE.rst
import collections.abc
import copy
import html
import logging
from os.path import split
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import matplotlib.pyplot as plt
from gammapy.maps import Map, RegionGeom
from gammapy.modeling import Covariance, Parameter, Parameters
from gammapy.modeling.covariance import CovarianceMixin
from gammapy.utils.scripts import from_yaml, make_name, make_path, to_yaml, write_yaml

__all__ = ["Model", "Models", "DatasetModels", "ModelBase"]


log = logging.getLogger(__name__)


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


def _get_model_class_from_dict(data):
    """Get a model class from a dictionary."""
    from . import (
        MODEL_REGISTRY,
        PRIOR_REGISTRY,
        SPATIAL_MODEL_REGISTRY,
        SPECTRAL_MODEL_REGISTRY,
        TEMPORAL_MODEL_REGISTRY,
    )

    if "type" in data:
        cls = MODEL_REGISTRY.get_cls(data["type"])
    elif "spatial" in data:
        cls = SPATIAL_MODEL_REGISTRY.get_cls(data["spatial"]["type"])
    elif "spectral" in data:
        cls = SPECTRAL_MODEL_REGISTRY.get_cls(data["spectral"]["type"])
    elif "temporal" in data:
        cls = TEMPORAL_MODEL_REGISTRY.get_cls(data["temporal"]["type"])
    elif "prior" in data:
        cls = PRIOR_REGISTRY.get_cls(data["prior"]["type"])
    return cls


def _build_parameters_from_dict(data, default_parameters):
    """Build a `~gammapy.modeling.Parameters` object from input dictionary and default parameter values."""
    par_data = []

    input_names = [_["name"] for _ in data]

    for par in default_parameters:
        par_dict = par.to_dict()
        try:
            index = input_names.index(par_dict["name"])
            par_dict.update(data[index])
        except ValueError:
            log.warning(
                f"Parameter '{par_dict['name']}' not defined in YAML file."
                f" Using default value: {par_dict['value']} {par_dict['unit']}"
            )
        par_data.append(par_dict)

    return Parameters.from_dict(par_data)


def _check_name_unique(model, names):
    """Check if a model is not duplicated"""
    if model.name in names:
        raise (
            ValueError(
                f"Model names must be unique. Models named '{model.name}' are duplicated."
            )
        )
    return


def _write_models(
    models,
    path,
    overwrite=False,
    full_output=False,
    overwrite_templates=False,
    write_covariance=True,
    checksum=False,
    extra_dict=None,
):
    """Write models to YAML file with additionnal informations using an `extra_dict`"""

    base_path, _ = split(path)
    path = make_path(path)
    base_path = make_path(base_path)

    if path.exists() and not overwrite:
        raise IOError(f"File exists already: {path}")

    if (
        write_covariance
        and models.covariance is not None
        and len(models.parameters) != 0
    ):
        filecovar = path.stem + "_covariance.dat"
        kwargs = dict(format="ascii.fixed_width", delimiter="|", overwrite=overwrite)
        models.write_covariance(base_path / filecovar, **kwargs)
        models._covar_file = filecovar

    yaml_str = ""
    if extra_dict is not None:
        yaml_str += to_yaml(extra_dict)
    yaml_str += models.to_yaml(full_output, overwrite_templates)

    write_yaml(yaml_str, path, overwrite=overwrite, checksum=checksum)


class ModelBase:
    """Model base class."""

    _type = None

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

        covariance_data = kwargs.get("covariance_data", None)

        if covariance_data is not None:
            self.covariance = covariance_data

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)

        if isinstance(value, Parameter):
            return value.__get__(self, None)

        return value

    @property
    def type(self):
        return self._type

    def __init_subclass__(cls, **kwargs):
        # Add parameters list on the model sub-class (not instances)
        cls.default_parameters = Parameters(
            [_ for _ in cls.__dict__.values() if isinstance(_, Parameter)]
        )

    @classmethod
    def from_parameters(cls, parameters, **kwargs):
        """Create model from parameter list.

        Parameters
        ----------
        parameters : `Parameters`
            Parameters for init.

        Returns
        -------
        model : `Model`
            Model instance.
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
            error = np.nan_to_num(par.error**2, nan=1)
            covar = Covariance(pars, data=[[error]])
            self._covariance.set_subcovariance(covar)

        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._check_covariance()
        self._covariance.data = covariance

        for par in self.parameters:
            pars = Parameters([par])
            variance = self._covariance.get_subcovariance(pars).data
            par.error = np.sqrt(variance[0][0])

    @property
    def parameters(self):
        """Parameters as a `~gammapy.modeling.Parameters` object."""
        return Parameters(
            [getattr(self, name) for name in self.default_parameters.names]
        )

    @property
    def parameters_unique_names(self):
        return self.parameters.unique_parameters.names

    def copy(self, **kwargs):
        """Deep copy."""
        return copy.deepcopy(self)

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
                    "interp",
                    "scale_method",
                    "is_norm",
                ]:
                    default = init[item]

                    if par[item] == default or (
                        np.isnan(par[item]) and np.isnan(default)
                    ):
                        del par[item]

                if par["frozen"] == init["frozen"]:
                    del par["frozen"]

                if init["unit"] == "":
                    del par["unit"]

        data = {"type": tag, "parameters": params}

        if self.type is None:
            return data
        else:
            return {self.type: data}

    @classmethod
    def from_dict(cls, data, **kwargs):
        key0 = next(iter(data))

        if key0 in ["spatial", "temporal", "spectral"]:
            data = data[key0]

        if data["type"] not in cls.tag:
            raise ValueError(
                f"Invalid model type {data['type']} for class {cls.__name__}"
            )

        parameters = _build_parameters_from_dict(
            data["parameters"], cls.default_parameters
        )

        return cls.from_parameters(parameters, **kwargs)

    def __str__(self):
        string = f"{self.__class__.__name__}\n"
        if len(self.parameters) > 0:
            string += f"\n{self.parameters.to_table()}"
        return string

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    @property
    def frozen(self):
        """Frozen status of a model, True if all parameters are frozen."""
        return np.all([p.frozen for p in self.parameters])

    def freeze(self):
        """Freeze all parameters."""
        self.parameters.freeze_all()

    def unfreeze(self):
        """Restore parameters frozen status to default."""
        for p, default in zip(self.parameters, self.default_parameters):
            p.frozen = default.frozen

    def reassign(self, datasets_names, new_datasets_names):
        """Reassign a model from one dataset to another.

        Parameters
        ----------
        datasets_names : str or list
            Name of the datasets where the model is currently defined.
        new_datasets_names : str or list
            Name of the datasets where the model should be defined instead.
            If multiple names are given the two list must have the save length,
            as the reassignment is element-wise.

        Returns
        -------
        model : `Model`
            Reassigned model.

        """
        model = self.copy(name=self.name)

        if not isinstance(datasets_names, list):
            datasets_names = [datasets_names]

        if not isinstance(new_datasets_names, list):
            new_datasets_names = [new_datasets_names]

        if isinstance(model.datasets_names, str):
            model.datasets_names = [model.datasets_names]

        if getattr(model, "datasets_names", None):
            for name, name_new in zip(datasets_names, new_datasets_names):
                model.datasets_names = [
                    _.replace(name, name_new) for _ in model.datasets_names
                ]

        return model


class Model:
    """Model class that contains only methods to create a model listed in the registries."""

    @staticmethod
    def create(tag, model_type=None, *args, **kwargs):
        """Create a model instance.

        Examples
        --------
        >>> from gammapy.modeling.models import Model
        >>> spectral_model = Model.create(
        ...            "pl-2", model_type="spectral", amplitude="1e-10 cm-2 s-1", index=3
        ...        )
        >>> type(spectral_model)
        <class 'gammapy.modeling.models.spectral.PowerLaw2SpectralModel'>
        """
        data = {"type": tag}
        if model_type is not None:
            data = {model_type: data}

        cls = _get_model_class_from_dict(data)
        return cls(*args, **kwargs)

    @staticmethod
    def from_dict(data):
        """Create a model instance from a dictionary."""
        cls = _get_model_class_from_dict(data)
        return cls.from_dict(data)


class DatasetModels(collections.abc.Sequence, CovarianceMixin):
    """Immutable models container.

    Parameters
    ----------
    models : `SkyModel`, list of `SkyModel` or `Models`
        Sky models.
    covariance_data : `~numpy.ndarray`
        Covariance data.
    """

    def __init__(self, models=None, covariance_data=None):
        if models is None:
            models = []

        if isinstance(models, (Models, DatasetModels)):
            if covariance_data is None and models.covariance is not None:
                covariance_data = models.covariance.data
            models = models._models
        elif isinstance(models, ModelBase):
            models = [models]
        elif not isinstance(models, list):
            raise TypeError(f"Invalid type: {models!r}")

        unique_names = []

        for model in models:
            _check_name_unique(model, names=unique_names)
            unique_names.append(model.name)

        self._models = models
        self._covar_file = None

        self._covariance = Covariance(self.parameters)

        # Set separately because this triggers the update mechanism on the sub-models
        if covariance_data is not None:
            self.covariance = covariance_data

    @property
    def parameters(self):
        """Parameters as a `~gammapy.modeling.Parameters` object."""
        return Parameters.from_stack([_.parameters for _ in self._models])

    @property
    def parameters_unique_names(self):
        """List of unique parameter names. Return formatted as model_name.par_type.par_name."""
        names = []
        for model in self._models:
            for par_name in model.parameters_unique_names:
                components = [model.name, par_name]
                name = ".".join(components)
                names.append(name)
        return names

    @property
    def names(self):
        """List of model names."""
        return [m.name for m in self._models]

    @classmethod
    def read(cls, filename, checksum=False):
        """Read from YAML file.

        Parameters
        ----------
        filename : str
            input filename
        checksum : bool, optional
            Whether to perform checksum verification. Default is False.
        """
        yaml_str = make_path(filename).read_text()
        path, filename = split(filename)
        return cls.from_yaml(yaml_str, path=path, checksum=checksum)

    @classmethod
    def from_yaml(cls, yaml_str, path="", checksum=False):
        """Create from YAML string.

        Parameters
        ----------
        yaml_str : str
            yaml str
        path : str
            base path of model files
        checksum : bool, optional
            Whether to perform checksum verification. Default is False.


        """
        data = from_yaml(yaml_str, checksum=checksum)
        # TODO : for now metadata are not kept. Add proper metadata creation.
        data.pop("metadata", None)
        return cls.from_dict(data, path=path)

    @classmethod
    def from_dict(cls, data, path=""):
        """Create from dictionary."""
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

    def write(
        self,
        path,
        overwrite=False,
        full_output=False,
        overwrite_templates=False,
        write_covariance=True,
        checksum=False,
    ):
        """Write to YAML file.

        Parameters
        ----------
        path : `pathlib.Path` or str
            Path to write files.
        overwrite : bool, optional
            Overwrite existing file. Default is False.
        full_output : bool, optional
            Store full parameter output. Default is False.
        overwrite_templates : bool, optional
            Overwrite templates FITS files. Default is False.
        write_covariance : bool, optional
            Whether to save the covariance. Default is True.
        checksum : bool, optional
            When True adds a CHECKSUM entry to the file.
            Default is False.
        """
        _write_models(
            self,
            path,
            overwrite,
            full_output,
            overwrite_templates,
            write_covariance,
            checksum,
        )

    def to_yaml(self, full_output=False, overwrite_templates=False):
        """Convert to YAML string.

        Parameters
        ----------
         full_output : bool, optional
            Store full parameter output. Default is False.
        overwrite_templates : bool, optional
            Overwrite templates FITS files. Default is False.
        """
        data = self.to_dict(full_output, overwrite_templates)
        return to_yaml(data)

    def update_link_label(self):
        """Update linked parameters labels used for serialisation and print."""
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

    def to_dict(self, full_output=False, overwrite_templates=False):
        """Convert to dictionary."""
        self.update_link_label()

        models_data = []
        for model in self._models:
            model_data = model.to_dict(full_output)
            models_data.append(model_data)
            if (
                hasattr(model, "spatial_model")
                and model.spatial_model is not None
                and "template" in model.spatial_model.tag
            ):
                model.spatial_model.write(overwrite=overwrite_templates)
            if model.tag == "TemplateNPredModel":
                model.write(overwrite=overwrite_templates)

        if self._covar_file is not None:
            return {
                "components": models_data,
                "covariance": str(self._covar_file),
            }
        else:
            return {"components": models_data}

    def to_parameters_table(self):
        """Convert model parameters to a `~astropy.table.Table`."""
        table = self.parameters.to_table()
        # Warning: splitting of parameters will break is source name has a "." in its name.
        model_name = [name.split(".")[0] for name in self.parameters_unique_names]
        table.add_column(model_name, name="model", index=0)
        return table

    def update_parameters_from_table(self, t):
        """Update models from a `~astropy.table.Table`."""
        parameters_dict = [dict(zip(t.colnames, row)) for row in t]
        for k, data in enumerate(parameters_dict):
            self.parameters[k].update_from_dict(data)

    def read_covariance(self, path, filename="_covariance.dat", **kwargs):
        """Read covariance data from file.

        Parameters
        ----------
        path : str or `Path`
            Base path.
        filename : str
            Filename.
        **kwargs : dict
            Keyword arguments passed to `~astropy.table.Table.read`.

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
        """Write covariance to file.

        Parameters
        ----------
        filename : str
            Filename.
        **kwargs : dict
            Keyword arguments passed to `~astropy.table.Table.write`.

        """
        names = self.parameters_unique_names
        table = Table()
        table["Parameters"] = names

        for idx, name in enumerate(names):
            values = self.covariance.data[idx]
            table[str(idx)] = values

        table.write(make_path(filename), **kwargs)

    def __str__(self):
        self.update_link_label()

        str_ = f"{self.__class__.__name__}\n\n"

        for idx, model in enumerate(self):
            str_ += f"Component {idx}: "
            str_ += str(model)

        return str_.expandtabs(tabsize=2)

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def __add__(self, other):
        if isinstance(other, (Models, list)):
            return Models([*self, *other])
        elif isinstance(other, ModelBase):
            _check_name_unique(other, self.names)
            return Models([*self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def __getitem__(self, key):
        if isinstance(key, np.ndarray) and key.dtype == bool:
            return self.__class__(list(np.array(self._models)[key]))
        else:
            return self._models[self.index(key)]

    def index(self, key):
        if isinstance(key, (int, slice)):
            return key
        elif isinstance(key, str):
            return self.names.index(key)
        elif isinstance(key, ModelBase):
            return self._models.index(key)
        else:
            raise TypeError(f"Invalid type: {type(key)!r}")

    def __len__(self):
        return len(self._models)

    def _ipython_key_completions_(self):
        return self.names

    def copy(self, copy_data=False):
        """Deep copy.

        Parameters
        ----------
        copy_data : bool
            Whether to copy data attached to template models.

        Returns
        -------
        models: `Models`
            Copied models.
        """
        models = []

        for model in self:
            model_copy = model.copy(name=model.name, copy_data=copy_data)
            models.append(model_copy)

        return self.__class__(
            models=models, covariance_data=self.covariance.data.copy()
        )

    def _slice_by_energy(self, energy_min, energy_max, sum_over_energy_groups=False):
        """Copy models and slice TemplateNPredModel in energy range.

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds of the slice
        sum_over_energy_groups : bool
            Whether to sum over the energy groups or not. Default is False.

        Returns
        -------
        models : `Models`
            Sliced models.
        """
        models_sliced = Models(self.copy())
        for k, m in enumerate(models_sliced):
            if m.tag == "TemplateNPredModel":
                m_sliced = m.slice_by_energy(energy_min, energy_max)
                if sum_over_energy_groups:
                    m_sliced.map = m_sliced.map.sum_over_axes(keepdims=True)
                models_sliced[k] = m_sliced
        return models_sliced

    def select(
        self,
        name_substring=None,
        datasets_names=None,
        tag=None,
        model_type=None,
        frozen=None,
    ):
        """Select models that meet all specified conditions.

        Parameters
        ----------
        name_substring : str, optional
            Substring contained in the model name. Default is None.
        datasets_names : str or list, optional
            Name of the dataset. Default is None.
        tag : str or list, optional
            Model tag. Default is None.
        model_type : {'None', 'spatial', 'spectral'}
           Type of model, used together with "tag", if the tag is not unique. Default is None.
        frozen : bool, optional
            If True, select models with all parameters frozen; if False, exclude them.  Default is None.

        Returns
        -------
        models : `DatasetModels`
            Selected models.
        """
        mask = self.selection_mask(
            name_substring, datasets_names, tag, model_type, frozen
        )
        return self[mask]

    def selection_mask(
        self,
        name_substring=None,
        datasets_names=None,
        tag=None,
        model_type=None,
        frozen=None,
    ):
        """Create a mask of models, that meet all specified conditions.

        Parameters
        ----------
        name_substring : str, optional
            Substring contained in the model name. Default is None.
        datasets_names : str or list of str, optional
            Name of the dataset. Default is None.
        tag : str or list of str, optional
            Model tag. Default is None.
        model_type : {'None', 'spatial', 'spectral'}
           Type of model, used together with "tag", if the tag is not unique. Default is None.
        frozen : bool, optional
            Select models with all parameters frozen if True, exclude them if False. Default is None.

        Returns
        -------
        mask : `numpy.array`
            Boolean mask, True for selected models.
        """
        selection = np.ones(len(self), dtype=bool)

        if tag and not isinstance(tag, list):
            tag = [tag]

        if datasets_names and not isinstance(datasets_names, list):
            datasets_names = [datasets_names]

        for idx, model in enumerate(self):
            if name_substring:
                selection[idx] &= name_substring in model.name

            if datasets_names:
                selection[idx] &= model.datasets_names is None or np.any(
                    [name in model.datasets_names for name in datasets_names]
                )

            if tag:
                if model_type is None:
                    sub_model = model
                else:
                    sub_model = getattr(model, f"{model_type}_model", None)

                if sub_model:
                    selection[idx] &= np.any([t in sub_model.tag for t in tag])
                else:
                    selection[idx] &= False

            if frozen is not None:
                if frozen:
                    selection[idx] &= model.frozen
                else:
                    selection[idx] &= ~model.frozen

        return np.array(selection, dtype=bool)

    def select_mask(self, mask, margin="0 deg", use_evaluation_region=True):
        """Check if sky models contribute within a mask map.

        Parameters
        ----------
        mask : `~gammapy.maps.WcsNDMap` of boolean type
            Map containing a boolean mask.
        margin : `~astropy.unit.Quantity`, optional
            Add a margin in degree to the source evaluation radius.
            Used to take into account PSF width. Default is "0 deg".
        use_evaluation_region : bool, optional
            Account for the extension of the model or not. Default is True.

        Returns
        -------
        models : `DatasetModels`
            Selected models contributing inside the region where mask==True.
        """
        models = []

        if not mask.geom.is_image:
            mask = mask.reduce_over_axes(func=np.logical_or)

        for model in self.select(tag="sky-model"):
            if use_evaluation_region:
                contributes = model.contributes(mask=mask, margin=margin)
            else:
                contributes = mask.get_by_coord(model.position, fill_value=0)

            if np.any(contributes):
                models.append(model)

        return self.__class__(models=models)

    def select_from_geom(self, geom, **kwargs):
        """Select models that fall inside a given geometry.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Geometry to select models from.
        **kwargs : dict
            Keyword arguments passed to `~gammapy.modeling.models.DatasetModels.select_mask`.

        Returns
        -------
        models : `DatasetModels`
            Selected models.
        """
        mask = Map.from_geom(geom=geom, data=True, dtype=bool)
        return self.select_mask(mask=mask, **kwargs)

    def select_region(self, regions, wcs=None):
        """Select sky models with center position contained within a given region.

        Parameters
        ----------
        regions : str, `~regions.Region` or list of `~regions.Region`
            Region or list of regions (pixel or sky regions accepted).
            A region can be defined as a string ind DS9 format as well.
            See http://ds9.si.edu/doc/ref/region.html for details.
        wcs : `~astropy.wcs.WCS`, optional
            World coordinate system transformation. Default is None.

        Returns
        -------
        models : `DatasetModels`
            Selected models.
        """
        geom = RegionGeom.from_regions(regions, wcs=wcs)

        models = []

        for model in self.select(tag="sky-model"):
            if geom.contains(model.position):
                models.append(model)

        return self.__class__(models=models)

    def restore_status(self, restore_values=True):
        """Context manager to restore status.

        A copy of the values is made on enter,
        and those values are restored on exit.

        Parameters
        ----------
        restore_values : bool, optional
            Restore values if True, otherwise restore only frozen status and covariance matrix.
            Default is True.

        """
        return restore_models_status(self, restore_values)

    def set_parameters_bounds(
        self, tag, model_type, parameters_names=None, min=None, max=None, value=None
    ):
        """Set bounds for the selected models types and parameters names.

        Parameters
        ----------
        tag : str or list
            Tag of the models.
        model_type : {"spatial", "spectral", "temporal"}
            Type of model.
        parameters_names : str or list, optional
            Parameters names. Default is None.
        min : float, optional
            Minimum value. Default is None.
        max : float, optional
            Maximum value. Default is None.
        value : float, optional
            Initial value. Default is None.
        """
        models = self.select(tag=tag, model_type=model_type)
        parameters = models.parameters.select(name=parameters_names, type=model_type)
        n = len(parameters)

        if min is not None:
            parameters.min = np.ones(n) * min
        if max is not None:
            parameters.max = np.ones(n) * max
        if value is not None:
            parameters.value = np.ones(n) * value

    def freeze(self, model_type=None):
        """Freeze parameters depending on model type.

        Parameters
        ----------
        model_type : {None, "spatial", "spectral"}
           Freeze all parameters or only spatial or only spectral.
           Default is None.
        """
        for m in self:
            m.freeze(model_type)

    def unfreeze(self, model_type=None):
        """Restore parameters frozen status to default depending on model type.

        Parameters
        ----------
        model_type : {None, "spatial", "spectral"}
           Restore frozen status to default for all parameters or only spatial or only spectral.
           Default is None.
        """
        for m in self:
            m.unfreeze(model_type)

    @property
    def frozen(self):
        """Boolean mask, True if all parameters of a given model are frozen."""
        return np.all([m.frozen for m in self])

    def reassign(self, dataset_name, new_dataset_name):
        """Reassign a model from one dataset to another.

        Parameters
        ----------
        dataset_name : str or list
            Name of the datasets where the model is currently defined.
        new_dataset_name : str or list
            Name of the datasets where the model should be defined instead.
            If multiple names are given the two list must have the save length,
            as the reassignment is element-wise.
        """
        models = [m.reassign(dataset_name, new_dataset_name) for m in self]
        return self.__class__(models)

    def to_template_sky_model(self, geom, spectral_model=None, name=None):
        """Merge a list of models into a single `~gammapy.modeling.models.SkyModel`.

        Parameters
        ----------
        geom : `Geom`
            Map geometry of the result template model.
        spectral_model : `~gammapy.modeling.models.SpectralModel`, optional
            One of the NormSpectralModel. Default is None.
        name : str, optional
            Name of the new model. Default is None.

        Returns
        -------
        model : `SkyModel`
            Template sky model.
        """
        from . import PowerLawNormSpectralModel, SkyModel, TemplateSpatialModel

        unit = u.Unit("1 / (cm2 s sr TeV)")
        map_ = Map.from_geom(geom, unit=unit)

        for m in self:
            map_ += m.evaluate_geom(geom).to(unit)

        spatial_model = TemplateSpatialModel(map_, normalize=False)

        if spectral_model is None:
            spectral_model = PowerLawNormSpectralModel()

        return SkyModel(
            spectral_model=spectral_model, spatial_model=spatial_model, name=name
        )

    def to_template_spectral_model(self, geom, mask=None):
        """Merge a list of models into a single `~gammapy.modeling.models.TemplateSpectralModel`.

        For each model the spatial component is integrated over the given geometry where the mask is true
        and multiplied by the spectral component value in each energy bin.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Map geometry on which the template model is computed.
        mask :  `~gammapy.maps.Map` with bool dtype.
            Evaluate the model only where the mask is True.

        Returns
        -------
        model : `~gammapy.modeling.models.TemplateSpectralModel`
            Template spectral model.
        """

        from . import TemplateSpectralModel

        energy = geom.axes[0].center
        if mask is None:
            mask = Map.from_geom(geom, data=True)
        elif mask.geom != geom:
            mask = mask.interp_to_geom(geom, method="nearest")
        values = 0
        for m in self:
            dnde = m.spectral_model(energy)
            spatial_integ = m.spatial_model.integrate_geom(geom)
            masked_integ = spatial_integ.data * np.isfinite(spatial_integ.data) * mask
            spatial_integ.data[~np.isfinite(spatial_integ.data)] = np.nan
            dnde *= masked_integ.sum(axis=(1, 2)) * spatial_integ.unit
            values += dnde
        return TemplateSpectralModel(energy, values)

    @property
    def positions(self):
        """Positions of the models as a `~astropy.coordinates.SkyCoord`."""
        positions = []

        for model in self.select(tag="sky-model"):
            if model.position:
                positions.append(model.position.icrs)
            else:
                log.warning(
                    f"Skipping model {model.name} - no spatial component present"
                )

        return SkyCoord(positions)

    def to_regions(self):
        """Return a list of the regions for the spatial models.

        Returns
        -------
        regions: list of `~regions.SkyRegion`
            Regions.
        """
        regions = []

        for model in self.select(tag="sky-model"):
            try:
                region = model.spatial_model.to_region()
                regions.append(region)
            except AttributeError:
                log.warning(
                    f"Skipping model {model.name} - no spatial component present"
                )
        return regions

    @property
    def wcs_geom(self):
        """Minimum WCS geom in which all the models are contained."""
        regions = self.to_regions()
        try:
            return RegionGeom.from_regions(regions).to_wcs_geom()
        except IndexError:
            log.error("No spatial component in any model. Geom not defined")

    def plot_regions(self, ax=None, kwargs_point=None, path_effect=None, **kwargs):
        """Plot extent of the spatial models on a given WCS axis.

        Parameters
        ----------
        ax : `~astropy.visualization.WCSAxes`, optional
            Axes to plot on. If no axes are given, an all-sky WCS
            is chosen using a CAR projection. Default is None.
        kwargs_point : dict, optional
            Keyword arguments passed to `~matplotlib.lines.Line2D` for plotting
            of point sources. Default is None.
        path_effect : `~matplotlib.patheffects.PathEffect`, optional
            Path effect applied to artists and lines. Default is None.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.artists.Artist`.


        Returns
        -------
        ax : `~astropy.visualization.WcsAxes`
            WCS axes.
        """
        regions = self.to_regions()

        geom = RegionGeom.from_regions(regions=regions)
        return geom.plot_region(
            ax=ax, kwargs_point=kwargs_point, path_effect=path_effect, **kwargs
        )

    def plot_positions(self, ax=None, **kwargs):
        """Plot the centers of the spatial models on a given WCS axis.

        Parameters
        ----------
        ax : `~astropy.visualization.WCSAxes`, optional
            Axes to plot on. If no axes are given, an all-sky WCS
            is chosen using a CAR projection. Default is None.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.scatter`.


        Returns
        -------
        ax : `~astropy.visualization.WcsAxes`
            WCS axes.
        """
        from astropy.visualization.wcsaxes import WCSAxes

        if ax is None or not isinstance(ax, WCSAxes):
            ax = Map.from_geom(self.wcs_geom).plot()

        kwargs.setdefault("marker", "*")
        kwargs.setdefault("color", "tab:blue")
        path_effects = kwargs.get("path_effects", None)

        xp, yp = self.positions.to_pixel(ax.wcs)
        p = ax.scatter(xp, yp, **kwargs)

        if path_effects:
            plt.setp(p, path_effects=path_effects)

        return ax


class Models(DatasetModels, collections.abc.MutableSequence):
    """Sky model collection.

    Parameters
    ----------
    models : `SkyModel`, list of `SkyModel` or `Models`
        Sky models.
    """

    def __delitem__(self, key):
        del self._models[self.index(key)]

    def __setitem__(self, key, model):
        from gammapy.modeling.models import (
            FoVBackgroundModel,
            SkyModel,
            TemplateNPredModel,
        )

        if isinstance(model, (SkyModel, FoVBackgroundModel, TemplateNPredModel)):
            self._models[self.index(key)] = model
        else:
            raise TypeError(f"Invalid type: {model!r}")

    def insert(self, idx, model):
        _check_name_unique(model, self.names)
        self._models.insert(idx, model)

    def set_prior(self, parameters, priors):
        for parameter, prior in zip(parameters, priors):
            parameter.prior = prior


class restore_models_status:
    def __init__(self, models, restore_values=True):
        self.restore_values = restore_values
        self.models = models
        self.values = [_.value for _ in models.parameters]
        self.frozen = [_.frozen for _ in models.parameters]
        self.covariance_data = models.covariance.data

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        for value, par, frozen in zip(self.values, self.models.parameters, self.frozen):
            if self.restore_values:
                par.value = value
            par.frozen = frozen
        self.models.covariance = self.covariance_data

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to serialize models."""
from gammapy.cube.fit import MapDataset, MapEvaluator
from .models import (
    BackgroundModel,
    SkyDiffuseCube,
    SkyModel,
    SkyModels,
)
from .parameter import Parameters

__all__ = ["models_to_dict", "dict_to_models", "dict_to_datasets", "datasets_to_dict"]


def models_to_dict(models, selection="all"):
    """Convert list of models to dict.

    Parameters
    ----------
    models : list
        Python list of Model objects
    selection : {"all", "simple"}
        Selection of information to include
    """
    # update shared parameters names for serialization
    _rename_shared_parameters(models)

    models_data = []
    for model in models:
        model_data = model.to_dict(selection)
        # De-duplicate if model appears several times
        if model_data not in models_data:
            models_data.append(model_data)

    # restore shared parameters names after serialization
    _restore_shared_parameters(models)

    return {"components": models_data}


def _rename_shared_parameters(models):
    params_list = []
    params_shared = []
    for model in models:
        for param in model.parameters:
            if param not in params_list:
                params_list.append(param)
            elif param not in params_shared:
                params_shared.append(param)
    for k, param in enumerate(params_shared):
        param.name = param.name + "@shared_" + str(k)


def _restore_shared_parameters(models):
    for model in models:
        for param in model.parameters:
            param.name = param.name.split("@")[0]


def dict_to_models(data, link=True):
    """De-serialise model data to Model objects.

    Parameters
    ----------
    data : dict
        Serialised model information
    link : bool
        check for shared parameters and link them
    """
    models = []
    for component in data["components"]:
        # background models are created separately
        if component["type"] == "BackgroundModel":
            model = BackgroundModel.from_dict(component)

        if component["type"] == "SkyDiffuseCube":
            model = SkyDiffuseCube.from_dict(component)

        if component["type"] == "SkyModel":
            model = SkyModel.from_dict(component)

        models.append(model)

    if link is True:
        _link_shared_parameters(models)
    return models


def _link_shared_parameters(models):
    shared_register = {}
    for model in models:
        for param in model.parameters:
            name = param.name
            if "@" in name:
                if name in shared_register:
                    new_param = shared_register[name]
                    ind = model.parameters.names.index(name)
                    model.parameters.parameters[ind] = new_param
                    if isinstance(model, SkyModel) is True:
                        spatial_params = model.spatial_model.parameters
                        spectral_params = model.spectral_model.parameters
                        if name in spatial_params.names:
                            ind = spatial_params.names.index(name)
                            spatial_params.parameters[ind] = new_param
                        elif name in spectral_params.names:
                            ind = spectral_params.names.index(name)
                            spectral_params.parameters[ind] = new_param
                else:
                    param.name = name.split("@")[0]
                    shared_register[name] = param


def datasets_to_dict(datasets, path, selection, overwrite):
    unique_models = []
    unique_backgrounds = []
    datasets_dictlist = []

    for dataset in datasets:
        filename = path + "data_" + dataset.name + ".fits"
        dataset.write(filename, overwrite)
        datasets_dictlist.append(dataset.to_dict(filename=filename))

        for model in dataset.model.skymodels:
            if model not in unique_models:
                unique_models.append(model)

        if dataset.background_model not in unique_backgrounds:
            unique_backgrounds.append(dataset.background_model)

    datasets_dict = {"datasets": datasets_dictlist}
    components_dict = models_to_dict(unique_models + unique_backgrounds, selection)
    return datasets_dict, components_dict


class dict_to_datasets:
    """add models and backgrounds to datasets

    Parameters
    ----------
    datasets : `~gammapy.modeling.Datasets`
        Datasets
    components : dict
        dict describing model components
    """

    def __init__(self, data_list, components):
        self.params_register = {}
        self.cube_register = {}

        self.models = dict_to_models(components, link=False)
        self.backgrounds = []
        self.datasets = []

        for data in data_list["datasets"]:
            dataset = MapDataset.read(data["filename"], name=data["name"])
            bkg_name = data["background"]
            model_names = data["models"]
            self.update_dataset(dataset, components, bkg_name, model_names)
            self.datasets.append(dataset)
        _link_shared_parameters(self.models + self.backgrounds)

    def update_dataset(self, dataset, components, model_names):

        backgrounds = []
        for component in components["components"]:
            if component["type"] == "BackgroundModel":
                background_model = self.add_background(dataset, component, bkg_prev)
                self.link_background_parameters(component, background_model)
                backgrounds.append(background_model)
                if background_model not in self.backgrounds:
                    self.backgrounds.append(background_model)

        dataset.background_model = BackgroundModels(backgrounds)
        models = [model for model in self.models if model.name in model_names]
        dataset.model = SkyModels(models)

    def add_background(self, dataset, component, bkg_prev):
        if component["name"].strip().upper() in bkg_prev:
            BGind = bkg_prev.index(component["name"].strip().upper())
        elif component["name"] in bkg_prev:
            BGind = bkg_prev.index(component["name"])
            background_model = dataset.background_model.models[BGind]
        background_model.name = component["name"]
        return background_model

    def link_background_parameters(self, component, background_model):
        """Link parameters to background."""
        try:
            params = self.params_register[component["name"]]
        except KeyError:
            params = Parameters.from_dict(component)
            self.params_register[component["name"]] = params
        background_model.parameters = params

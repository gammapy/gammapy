# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to serialize models."""
from gammapy.cube.fit import MapDataset, MapEvaluator
from gammapy.cube.models import (
    BackgroundModel,
    BackgroundModels,
    SkyDiffuseCube,
    SkyModel,
    SkyModels,
)
from gammapy.image.models import SPATIAL_MODELS
from gammapy.spectrum.models import SPECTRAL_MODELS
from gammapy.utils.fitting import Parameters

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
        model_data = _model_to_dict(model, selection)
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


def _model_to_dict(model, selection):
    data = {}
    data["name"] = model.name
    if getattr(model, "filename", None) is not None:
        data["filename"] = model.filename
    if model.__class__.__name__ == "SkyModel":
        data["spatial"] = model.spatial_model.to_dict(selection)
        if getattr(model.spatial_model, "filename", None) is not None:
            data["spatial"]["filename"] = model.spatial_model.filename
        data["spectral"] = model.spectral_model.to_dict(selection)
    else:
        data["model"] = model.to_dict(selection)

    return data


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
    for model in data["components"]:
        if "model" in model:
            if model["model"]["type"] == "BackgroundModel":
                continue
            else:
                raise NotImplementedError

        model = _dict_to_skymodel(model)
        models.append(model)
    if link is True:
        _link_shared_parameters(models)
    return models


def _dict_to_skymodel(model):
    if "spatial" in model and model["spatial"]["type"] in SPATIAL_MODELS:
        model_class = SPATIAL_MODELS[model["spatial"]["type"]]
        spatial_model = model_class.from_dict(model["spatial"])
    else:
        spatial_model = None
    if "spectral" in model and model["spectral"]["type"] in SPECTRAL_MODELS:
        model_class = SPECTRAL_MODELS[(model["spectral"]["type"])]
        spectral_model = model_class.from_dict(model["spectral"])
    else:
        spectral_model = None
    #    TODO add temporal model to SkyModel once applicable
    #    if "temporal" in model and model["temporal"]["type"] in temporal_models:
    #        model_class= tenporal_models[(model["tenporal"]["type"])]
    #        temporal_model = model_class.from_dict( model["temporal"])
    #    else:
    #        temporal_model = None
    return SkyModel(
        name=model["name"], spatial_model=spatial_model, spectral_model=spectral_model
    )


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
    from gammapy.utils.serialization import models_to_dict
    from gammapy.cube.models import BackgroundModels

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

        if isinstance(dataset.background_model, BackgroundModels):
            backgrounds = dataset.background_model.models
        else:
            backgrounds = [dataset.background_model]

        for background in backgrounds:
            if background not in unique_backgrounds:
                unique_backgrounds.append(background)

    datasets_dict = {"datasets": datasets_dictlist}
    components_dict = models_to_dict(unique_models + unique_backgrounds, selection)
    return datasets_dict, components_dict


class dict_to_datasets:
    """add models and backgrounds to datasets

    Parameters
    ----------
    datasets : `~gammapy.utils.fitting.Datasets`
        Datasets
    components : dict
        dict describing model components
    get_lists : bool
        get the datasets, models and backgrounds lists separetely (used to initialize FitManager)

    """

    def __init__(self, data_list, components):
        self.params_register = {}
        self.cube_register = {}

        self.models = dict_to_models(components, link=False)
        self.backgrounds = []
        self.datasets = []
        for data in data_list["datasets"]:
            dataset = MapDataset.read(data["filename"], name=data["name"])
            bkg_names = data["backgrounds"]
            model_names = data["models"]
            self.update_dataset(dataset, components, bkg_names, model_names)
            self.datasets.append(dataset)
        _link_shared_parameters(self.models + self.backgrounds)

    def update_dataset(self, dataset, components, bkg_names, model_names):
        if not isinstance(dataset.background_model, BackgroundModels):
            dataset.background_model = BackgroundModels([dataset.background_model])
        # TODO: remove isinstance checks once #2102  is resolved
        bkg_prev = [model.name for model in dataset.background_model.models]
        backgrounds = []
        for component in components["components"]:
            if (
                "model" in component
                and component["model"]["type"] == "BackgroundModel"
                and component["name"] in bkg_names
            ):
                background_model = self.add_background(dataset, component, bkg_prev)
                self.link_background_parameters(component, background_model)
                backgrounds.append(background_model)
                if background_model not in self.backgrounds:
                    self.backgrounds.append(background_model)

        dataset.background_model = BackgroundModels(backgrounds)
        models = [model for model in self.models if model.name in model_names]
        dataset.model = SkyModels(models)

    def add_background(self, dataset, component, bkg_prev):
        if "filename" in component:
            # check if file is already loaded in memory else read
            try:
                cube = self.cube_register[component["name"]]
            except KeyError:
                cube = SkyDiffuseCube.read(component["filename"])
                self.cube_register[component["name"]] = cube

            evaluator = MapEvaluator(model=cube, exposure=dataset.exposure, psf=dataset.psf, edisp=dataset.edisp)
            background_model = BackgroundModel(evaluator.compute_npred())
        else:
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
            params = Parameters.from_dict(component["model"])
            self.params_register[component["name"]] = params
        background_model.parameters = params

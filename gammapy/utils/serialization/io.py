# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to serialize models."""
import astropy.units as u
from gammapy.image import models as spatial
from gammapy.spectrum import models as spectral
from gammapy.cube.models import (
    SkyModel,
    SkyModels,
    SkyDiffuseCube,
    BackgroundModel,
    BackgroundModels,
)
from gammapy.utils.fitting import Parameters

__all__ = ["models_to_dict", "dict_to_models", "models_to_datasets"]


def models_to_dict(models, selection="all"):
    """Convert list of models to dict.

    Parameters
    -----------
    models : list
        Python list of Model objects
    selection : {"all", "simple"}
        Selection of information to include
    """
    models_data = []
    for model in models:
        model_data = _model_to_dict(model, selection)

        # De-duplicate if model appears several times
        if model_data not in models_data:
            models_data.append(model_data)

    return {"components": models_data}


def _model_to_dict(model, selection):
    data = {}
    data["name"] = getattr(model, "name", model.__class__.__name__)
    try:
        data["id"] = model.obs_id
    except AttributeError:
        pass
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


def dict_to_models(data):
    """De-serialise model data to Model objects.

    Parameters
    -----------
    data : dict
        Serialised model information
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

    return models


def _dict_to_skymodel(model):
    item = model["spatial"]
    if "filename" in item:
        spatial_model = getattr(spatial, item["type"]).read(item["filename"])
        spatial_model.filename = item["filename"]
        spatial_model.parameters = Parameters.from_dict(item)
    else:
        params = {x["name"]: x["value"] * u.Unit(x["unit"]) for x in item["parameters"]}
        spatial_model = getattr(spatial, item["type"])(**params)
        spatial_model.parameters = Parameters.from_dict(item)

    item = model["spectral"]
    if "energy" in item:
        energy = u.Quantity(item["energy"]["data"], item["energy"]["unit"])
        values = u.Quantity(item["values"]["data"], item["values"]["unit"])
        params = {"energy": energy, "values": values}
        spectral_model = getattr(spectral, item["type"])(**params)
        spectral_model.parameters = Parameters.from_dict(item)
    else:
        params = {x["name"]: x["value"] * u.Unit(x["unit"]) for x in item["parameters"]}
        spectral_model = getattr(spectral, item["type"])(**params)
        spectral_model.parameters = Parameters.from_dict(item)

    return SkyModel(
        name=model["name"], spatial_model=spatial_model, spectral_model=spectral_model
    )


def _get_backgrounds_names(dataset):
    BGnames = []
    for background in dataset.background_model.models:
        BGnames.append(background.name)
    return BGnames


def models_to_datasets(datasets, components, get_lists=False):
    """add models and background to datasets
    
    Parameters
    ----------
    datasets : object
        '~gammapy.utils,fitting.MapDatasets'
    components : dict
        dict describing model components
    get_lists : bool
        get the datasets, models and backgrounds lists separetely (used to initialize FitManager)
        
    """
    datasets_list = datasets.datasets
    models = dict_to_models(components)

    params_register = {}
    cube_register = {}
    backgrounds_local = []
    backgrounds_global = []
    for dataset in datasets_list:

        if not isinstance(dataset.background_model, BackgroundModels):
            dataset.background_model = BackgroundModels[dataset.background_model]
        BGnames = _get_backgrounds_names(dataset)

        backgrounds = []
        for component in components["components"]:
            if (
                "model" in component
                and component["model"]["type"] == "BackgroundModel"
                and component["id"] in ["global", "local", dataset.obs_id]
            ):
                if "filename" in component:
                    # check if file is already loaded in memory else read
                    try:
                        cube = cube_register[component["name"]]
                    except KeyError:
                        cube = SkyDiffuseCube.read(component["filename"])
                        cube_register[component["name"]] = cube
                    background_model = BackgroundModel.from_skymodel(
                        cube,
                        exposure=dataset.exposure,
                        psf=dataset.psf,
                        edisp=dataset.edisp,
                    )
                else:
                    if component["name"].strip().upper() in BGnames:
                        BGind = BGnames.index(component["name"].strip().upper())
                    elif component["name"] in BGnames:
                        BGind = BGnames.index(component["name"])
                    else:
                        raise ValueError("Unknown Background")
                    background_model = dataset.background_model.models[BGind]
                background_model.name = component["name"]

                # link parameters for global backgrounds
                if component["id"] == "global":
                    try:
                        params = params_register[component["name"]]
                    except KeyError:
                        params = Parameters.from_dict(component["model"])
                        params_register[component["name"]] = params
                    background_model.parameters = params
                    background_model.obs_id = "global"
                    backgrounds_global.append(background_model)
                elif component["id"] in ["local", dataset.obs_id]:
                    background_model.parameters = Parameters.from_dict(
                        component["model"]
                    )
                    background_model.obs_id = dataset.obs_id
                    backgrounds_local.append(background_model)

                backgrounds.append(background_model)

        dataset.background_model = BackgroundModels(backgrounds)
        dataset.model = SkyModels(models)

    if get_lists is True:
        return datasets_list, models, backgrounds_global, backgrounds_local

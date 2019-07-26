# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to serialize models."""
import astropy.units as u
from ...image import models as spatial
from ...spectrum import models as spectral
from ...cube.models import SkyModel
from ..fitting import Parameters

__all__ = ["models_to_dict", "dict_to_models"]


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

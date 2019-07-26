# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to serialize models."""
import astropy.units as u
from ...image import models as spatial
from ...spectrum import models as spectral
from ...cube.models import SkyModel
from ..scripts import read_yaml
from ..fitting import Parameters


def models_to_dict(components_list, selection="all"):
    """ 
    Convert a list of models to dict

    Parameters
    -----------
    components_list : list of models
    selection : {"all", "simple"}
        Selection of information to include
    """
    dict_list=[]
    for kc in components_list:
        tmp_dict = {}
        tmp_dict["name"] = getattr(kc, "name", kc.__class__.__name__)
        try:
            tmp_dict["id"] = kc.obs_id
        except AttributeError:
            pass
        if getattr(kc, "filename", None) is not None:
            tmp_dict["filename"] = kc.filename

        if kc.__class__.__name__ == "SkyModel":
            tmp_dict["spatial"] = kc.spatial_model.to_dict(selection)
            if getattr(kc.spatial_model, "filename", None) is not None:
                tmp_dict["spatial"]["filename"] = kc.spatial_model.filename
            tmp_dict["spectral"] = kc.spectral_model.to_dict(selection)
        else:
            tmp_dict["model"] = kc.to_dict(selection)
        if tmp_dict not in dict_list:
            dict_list.append(tmp_dict)

    return {"components": dict_list}


def dict_to_models(filemodel):
    """ 
    Build SkyModels from yaml model file

    Parameters
    -----------
    filemodel :  filepath to yaml model file
    """
    components_list = read_yaml(filemodel)["components"]
    models_list = []
    for kc in components_list:
        keys = list(kc.keys())
        if "spatial" in keys and "spectral" in keys:
            item = kc["spatial"]
            if "filename" in list(item.keys()):
                spatial_model = getattr(spatial, item["type"]).read(item["filename"])
                spatial_model.filename = item["filename"]
                spatial_model.parameters = Parameters.from_dict(item)
            else:
                params = {
                    x["name"]: x["value"] * u.Unit(x["unit"]) for x in item["parameters"]
                }
                spatial_model = getattr(spatial, item["type"])(**params)
                spatial_model.parameters = Parameters.from_dict(item)
            item = kc["spectral"]
            if "energy" in list(item.keys()):
                energy = u.Quantity(item["energy"]["data"], item["energy"]["unit"])
                values = u.Quantity(item["values"]["data"], item["values"]["unit"])
                params = {"energy": energy, "values": values}
                spectral_model = getattr(spectral, item["type"])(**params)
                spectral_model.parameters = Parameters.from_dict(item)
            else:
                params = {
                    x["name"]: x["value"] * u.Unit(x["unit"]) for x in item["parameters"]
                }
                spectral_model = getattr(spectral, item["type"])(**params)
                spectral_model.parameters = Parameters.from_dict(item)
            models_list.append(
                SkyModel(
                    name=kc["name"],
                    spatial_model=spatial_model,
                    spectral_model=spectral_model,
                )
            )

    return models_list

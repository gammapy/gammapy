# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to serialize models."""
import astropy.units as u
from ...image import models as spatial
from ...spectrum import models as spectral
from ...cube.models import SkyModel
from ..scripts import read_yaml
from ..fitting import Parameters


def models_to_dict(components_list, selection="all"):
    """TODO: describe"""
    dict_list = []
    for kc in components_list:
        tmp_dict = {}
        tmp_dict["Name"] = getattr(kc, "name", kc.__class__.__name__)
        try:
            tmp_dict["ID"] = kc.obs_id
        except AttributeError:
            pass
        if getattr(kc, "filename", None) is not None:
            tmp_dict["Filename"] = kc.filename

        if kc.__class__.__name__ == "SkyModel":
            tmp_dict["SkySpatialModel"] = kc.spatial_model.to_dict(selection)
            if getattr(kc.spatial_model, "filename", None) is not None:
                tmp_dict["SkySpatialModel"]["Filename"] = kc.spatial_model.filename
            tmp_dict["SpectralModel"] = kc.spectral_model.to_dict(selection)
        else:
            tmp_dict["Model"] = kc.to_dict(selection)
        if tmp_dict not in dict_list:
            dict_list.append(tmp_dict)

    return {"Components": dict_list}


def dict_to_models(filemodel):
    """TODO: describe"""
    components_list = read_yaml(filemodel)["Components"]
    models_list = []
    for kc in components_list:
        keys = list(kc.keys())
        if "SkySpatialModel" in keys and "SpectralModel" in keys:
            item = kc["SkySpatialModel"]
            if "Filename" in list(item.keys()):
                spatial_model = getattr(spatial, item["Type"]).read(item["Filename"])
                spatial_model.filename = item["Filename"]
                spatial_model.parameters = Parameters.from_dict(item)
            else:
                params = {
                    x["name"]: x["value"] * u.Unit(x["unit"]) for x in item["Parameters"]
                }
                spatial_model = getattr(spatial, item["Type"])(**params)
                spatial_model.parameters = Parameters.from_dict(item)
            item = kc["SpectralModel"]
            if "Energy" in list(item.keys()):
                energy = u.Quantity(item["Energy"]["data"], item["Energy"]["unit"])
                values = u.Quantity(item["Values"]["data"], item["Values"]["unit"])
                params = {"energy": energy, "values": values}
                spectral_model = getattr(spectral, item["Type"])(**params)
                spectral_model.parameters = Parameters.from_dict(item)
            else:
                params = {
                    x["name"]: x["value"] * u.Unit(x["unit"]) for x in item["Parameters"]
                }
                spectral_model = getattr(spectral, item["Type"])(**params)
                spectral_model.parameters = Parameters.from_dict(item)
            models_list.append(
                SkyModel(
                    name=kc["Name"],
                    spatial_model=spatial_model,
                    spectral_model=spectral_model,
                )
            )

    return models_list

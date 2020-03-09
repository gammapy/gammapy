# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to serialize models."""
from .cube import SkyModel
from gammapy.utils.scripts import make_name

__all__ = ["models_to_dict", "dict_to_models"]


def models_to_dict(models):
    """Convert list of models to dict.

    Parameters
    ----------
    models : list
        Python list of Model objects
    """

    # De-duplicate if model appears several times
    unique_models = []
    for model in models:
        if model not in unique_models:
            unique_models.append(model)

    # update shared parameters refererence
    _update_link_reference(unique_models)

    models_data = []
    for model in unique_models:
        model_data = model.to_dict()
        models_data.append(model_data)

    return {"components": models_data}


def _update_link_reference(models):
    params_list = []
    params_shared = []
    for model in models:
        for param in model.parameters:
            if param not in params_list:
                params_list.append(param)
            elif param not in params_shared:
                params_shared.append(param)
    for param in params_shared:
        param._link_label_io = param.name + "@" + make_name()


def dict_to_models(data, link=True):
    """De-serialise model data to Model objects.

    Parameters
    ----------
    data : dict
        Serialised model information
    link : bool
        check for shared parameters and link them
    """
    from . import MODELS

    models = []
    for component in data["components"]:
        model = MODELS.get_cls(component["type"]).from_dict(component)
        models.append(model)

    if link:
        _link_shared_parameters(models)
    return models


def _link_shared_parameters(models):
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

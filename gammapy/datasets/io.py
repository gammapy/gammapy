# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.modeling.serialize import dict_to_models, models_to_dict


def datasets_to_dict(datasets, path, prefix, overwrite):
    """Convert datasets to dicts for serialization.

    Parameters
    ----------
    datasets : `~gammapy.modeling.Datasets`
        Datasets
    path : `pathlib.Path`
        path to write files
    prefix : str
        common prefix of file names
    overwrite : bool
        overwrite datasets FITS files
    """
    unique_models = []
    unique_backgrounds = []
    datasets_dictlist = []

    for dataset in datasets:
        filename = path / f"{prefix}_data_{dataset.name}.fits"
        dataset.write(filename, overwrite)
        datasets_dictlist.append(dataset.to_dict(filename=filename))

        if dataset.models is not None:
            for model in dataset.models:
                if model not in unique_models:
                    unique_models.append(model)

        try:
            if dataset.background_model not in unique_backgrounds:
                unique_backgrounds.append(dataset.background_model)
        except AttributeError:
            pass

    datasets_dict = {"datasets": datasets_dictlist}
    components_dict = models_to_dict(unique_models + unique_backgrounds)
    return datasets_dict, components_dict


def dict_to_datasets(data_list, components):
    """add models and backgrounds to datasets

    Parameters
    ----------
    datasets : `~gammapy.modeling.Datasets`
        Datasets
    components : dict
        dict describing model components
    """
    from . import DATASETS

    models = dict_to_models(components)
    datasets = []

    for data in data_list["datasets"]:
        dataset = DATASETS.get_cls(data["type"]).from_dict(data, components, models)
        datasets.append(dataset)
    return datasets

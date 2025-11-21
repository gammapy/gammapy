# Licensed under a 3-clause BSD style license - see LICENSE.rst


def _parse_datasets(datasets):
    """Parser used by Fit and Sampler classes."""
    from gammapy.datasets import Dataset, Datasets

    if isinstance(datasets, (list, Dataset)):
        datasets = Datasets(datasets)
    return datasets, datasets.parameters

import numpy as np
from astropy.utils import lazyproperty
from .parameter import Parameters


class Datasets:
    """Join multiple datasets

    Parameters
    ----------
    datasets : `Dataset` or list of `Dataset`
        List of `Dataset` objects ot be joined.
    mask : `~numpy.ndarray`
        Global fitting mask used for all datasets.

    """
    def __init__(self, datasets, mask=None):
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.datasets = datasets

        if mask is not None and not self.is_all_same_type:
            raise ValueError("Cannot apply mask if datasets are not of the same type.")

        self.mask = mask

        # join parameter lists
        parameters = []
        for dataset in self.datasets:
            parameters += dataset.parameters.parameters
        self.parameters = Parameters(parameters)

    @property
    def model(self):
        return self.datasets[0].model

    @lazyproperty
    def is_all_same_type(self):
        """Whether all contained datasets are of the same type"""
        dataset_type = type(self.datasets[0])
        is_dataset_type = [isinstance(_, dataset_type) for _ in self.datasets]
        return np.all(is_dataset_type)

    def likelihood(self, parameters=None):
        """Compute joint likelihood"""
        total_likelihood = 0
        # TODO: add parallel evaluation of likelihoods
        for dataset in self.datasets:
            total_likelihood += dataset.likelihood(parameters=parameters, mask=self.mask)
        return total_likelihood

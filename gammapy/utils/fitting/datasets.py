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

    @lazyproperty
    def types(self):
        """Types of the contained dataets"""
        return [type(dataset).__name__ for dataset in self.datasets]

    @lazyproperty
    def is_all_same_type(self):
        """Whether all contained datasets are of the same type"""
        return np.all(np.array(self.types) == self.types[0])

    def likelihood(self, parameters=None):
        """Compute joint likelihood"""
        total_likelihood = 0
        # TODO: add parallel evaluation of likelihoods
        for dataset in self.datasets:
            total_likelihood += dataset.likelihood(parameters=parameters, mask=self.mask)
        return total_likelihood

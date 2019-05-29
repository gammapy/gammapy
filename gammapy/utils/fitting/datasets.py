# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import copy
from collections import Counter
import numpy as np
from astropy.utils import lazyproperty
from .parameter import Parameters

__all__ = ["Dataset", "Datasets"]


class Dataset(abc.ABC):
    """Dataset abstract base class.

    TODO: add tutorial how to create your own dataset types.

    For now, see existing examples in Gammapy how this works:

    - `gammapy.cube.MapDataset`
    - `gammapy.spectrum.SpectrumDataset`
    - `gammapy.spectrum.FluxPointsDataset`
    """

    @abc.abstractmethod
    def likelihood(self):
        """Total likelihood (float, sum over bins).

        TODO: fix interface.
        Remove ``parameters`` and ``mask``, add update method?
        """
        pass

    @abc.abstractmethod
    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)


class Datasets:
    """Join multiple datasets.

    Parameters
    ----------
    datasets : `Dataset` or list of `Dataset`
        List of `Dataset` objects ot be joined.
    """

    def __init__(self, datasets):
        if not isinstance(datasets, list):
            datasets = [datasets]
        self._datasets = datasets

    @lazyproperty
    def parameters(self):
        # join parameter lists
        parameters = []
        for dataset in self.datasets:
            parameters += dataset.parameters.parameters
        return Parameters(parameters)

    @property
    def datasets(self):
        """List of datasets"""
        return self._datasets

    @property
    def types(self):
        """Types of the contained datasets"""
        return [type(dataset).__name__ for dataset in self.datasets]

    @property
    def is_all_same_type(self):
        """Whether all contained datasets are of the same type"""
        return np.all(np.array(self.types) == self.types[0])

    @property
    def is_all_same_shape(self):
        """Whether all contained datasets have the same data shape"""
        ref_shape = self.datasets[0].data_shape
        is_ref_shape = [dataset.data_shape == ref_shape for dataset in self.datasets]
        return np.all(is_ref_shape)

    def likelihood(self):
        """Compute joint likelihood"""
        total_likelihood = 0
        # TODO: add parallel evaluation of likelihoods
        for dataset in self.datasets:
            total_likelihood += dataset.likelihood()
        return total_likelihood

    def __str__(self):
        str_ = self.__class__.__name__ + "\n"
        str_ += "--------\n\n"

        counter = Counter(self.types)

        for key, value in counter.items():
            str_ += "\t{key}: {value} \n".format(key=key, value=value)

        return str_

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import copy
from collections import Counter
import numpy as np
from astropy.utils import lazyproperty
from gammapy.utils.scripts import read_yaml, write_yaml
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

    _residuals_labels = {
        "diff": "data - model",
        "diff/model": "(data - model) / model",
        "diff/sqrt(model)": "(data - model) / sqrt(model)",
    }

    @property
    def mask(self):
        """Combined fit and safe mask"""
        if self.mask_safe is not None and self.mask_fit is not None:
            mask = self.mask_safe & self.mask_fit
        elif self.mask_fit is not None:
            mask = self.mask_fit
        elif self.mask_safe is not None:
            mask = self.mask_safe
        else:
            mask = None
        return mask

    def likelihood(self):
        """Total likelihood given the current model parameters.
        """
        stat = self.likelihood_per_bin()

        if self.mask is not None:
            stat = stat[self.mask]

        return np.sum(stat, dtype=np.float64)

    @abc.abstractmethod
    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    @staticmethod
    def _compute_residuals(data, model, method="diff"):
        with np.errstate(invalid="ignore"):
            if method == "diff":
                residuals = data - model
            elif method == "diff/model":
                residuals = (data - model) / model
            elif method == "diff/sqrt(model)":
                residuals = (data - model) / np.sqrt(model)
            else:
                raise AttributeError(
                    f"Invalid method: {method!r}. Choose between 'diff',"
                    " 'diff/model' and 'diff/sqrt(model)'"
                )
        return residuals


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

    @property
    def parameters(self):
        # join parameter lists
        parameters = []
        for dataset in self.datasets:
            parameters += dataset.parameters.parameters
        return Parameters(parameters)

    @property
    def names(self):
        """List of dataset names"""
        return [_.name for _ in self.datasets]

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
            str_ += f"\t{key}: {value} \n"

        return str_

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    @classmethod
    def from_yaml(cls, filedata, filemodel):
        """De-serialize datasets from YAML and FITS files.

        Parameters
        ----------
        filedata : str
            filepath to yaml datasets file
        filemodel : str
            filepath to yaml models file

        Returns
        -------
        dataset : 'gammapy.modeling.Datasets'
            Datasets
        """
        from .serialize import dict_to_datasets

        components = read_yaml(filemodel)
        data_list = read_yaml(filedata)
        constructor = dict_to_datasets(data_list, components)
        return cls(constructor.datasets)

    def to_yaml(self, path, overwrite=False):
        """Serialize datasets to YAML and FITS files.

        Parameters
        ----------
        path : str
            path to write files
        selection : {"all", "simple"}
            "simple" option reduce models parameters attributes displayed to only
            name, value, unit,frozen
        """
        from .serialize import datasets_to_dict

        datasets_dict, components_dict = datasets_to_dict(
            self.datasets, path, overwrite
        )
        write_yaml(datasets_dict, path + "datasets.yaml", sort_keys=False)
        write_yaml(components_dict, path + "models.yaml", sort_keys=False)

    def stack_reduce(self):
        """Reduce the Datasets to a unique Dataset by stacking them together.

        This works only if all Dataset are of the same type and if a proper
        in-place stack method exists for the Dataset type.

        Returns
        -------
        dataset : ~gammapy.utils.Dataset
            the stacked dataset
        """
        if not self.is_all_same_type:
            raise ValueError(
                "Stacking impossible: all Datasets contained are not of a unique type."
            )

        dataset = self.datasets[0].copy()
        for ds in self.datasets[1:]:
            dataset.stack(ds)
        return dataset

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self.names.index(item)
        return self.datasets[item]

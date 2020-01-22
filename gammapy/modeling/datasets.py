# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import collections.abc
import copy
from warnings import warn
import numpy as np
from gammapy.utils.scripts import make_name, make_path, read_yaml, write_yaml
from gammapy.utils.table import table_from_row_data
from ..maps import WcsNDMap
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
        mask_safe = (
            self.mask_safe.data
            if isinstance(self.mask_safe, WcsNDMap)
            else self.mask_safe
        )
        mask_fit = (
            self.mask_fit.data if isinstance(self.mask_fit, WcsNDMap) else self.mask_fit
        )
        if mask_safe is not None and mask_fit is not None:
            mask = mask_safe & mask_fit
        elif mask_fit is not None:
            mask = mask_fit
        elif mask_safe is not None:
            mask = mask_safe
        else:
            mask = None
        return mask

    def stat_sum(self):
        """Total statistic given the current model parameters."""
        stat = self.stat_array()

        if self.mask is not None:
            stat = stat[self.mask]

        return np.sum(stat, dtype=np.float64)

    @abc.abstractmethod
    def stat_array(self):
        """Statistic array, one value per data point."""

    def copy(self, name=None):
        """A deep copy."""
        new = copy.deepcopy(self)
        if name is None:
            new.name = make_name()
        else:
            new.name = name
        return new

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


class Datasets(collections.abc.Sequence):
    """Dataset collection.

    Parameters
    ----------
    datasets : `Dataset` or list of `Dataset`
        Datasets
    """

    def __init__(self, datasets):
        if isinstance(datasets, Datasets):
            datasets = list(datasets)
        elif isinstance(datasets, list):
            dataset_list = []
            for data in datasets:
                if isinstance(data, Datasets):
                    dataset_list + list(data)
                elif isinstance(data, Dataset):
                    dataset_list.append(data)
                else:
                    raise TypeError(f"Invalid type: {datasets!r}")
        else:
            raise TypeError(f"Invalid type: {datasets!r}")

        unique_names = []
        renamed = False
        for dataset in dataset_list:
            while dataset.name in unique_names:
                dataset.name = make_name()  # replace duplicate
                if renamed is False:
                    warn("Dataset names must be unique, auto-replaced duplicates")
                    renamed = True  # avoid repetition
            unique_names.append(dataset.name)

        self._datasets = datasets

    @property
    def parameters(self):
        """Unique parameters (`~gammapy.modeling.Parameters`).

        Duplicate parameter objects have been removed.
        The order of the unique parameters remains.
        """
        parameters = Parameters.from_stack(_.parameters for _ in self)
        return parameters.unique_parameters

    @property
    def is_all_same_type(self):
        """Whether all contained datasets are of the same type"""
        return len(set(_.__class__ for _ in self)) == 1

    @property
    def is_all_same_shape(self):
        """Whether all contained datasets have the same data shape"""
        return len(set(_.data_shape for _ in self)) == 1

    def stat_sum(self):
        """Compute joint likelihood"""
        stat_sum = 0
        # TODO: add parallel evaluation of likelihoods
        for dataset in self:
            stat_sum += dataset.stat_sum()
        return stat_sum

    def __str__(self):
        str_ = self.__class__.__name__ + "\n"
        str_ += "--------\n"

        for idx, dataset in enumerate(self):
            str_ += f"idx={idx}, id={hex(id(dataset))!r}, name={dataset.name!r}\n"

        return str_

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    @classmethod
    def read(cls, filedata, filemodel):
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

        components = read_yaml(make_path(filemodel))
        data_list = read_yaml(make_path(filedata))
        datasets = dict_to_datasets(data_list, components)
        return cls(datasets)

    def write(self, path, prefix="", overwrite=False):
        """Serialize datasets to YAML and FITS files.

        Parameters
        ----------
        path : `pathlib.Path`
            path to write files
        prefix : str
            common prefix of file names
        overwrite : bool
            overwrite datasets FITS files
        """
        from .serialize import datasets_to_dict

        path = make_path(path)

        datasets_dict, components_dict = datasets_to_dict(self, path, prefix, overwrite)
        write_yaml(datasets_dict, path / f"{prefix}_datasets.yaml", sort_keys=False)
        write_yaml(components_dict, path / f"{prefix}_models.yaml", sort_keys=False)

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

        dataset = self[0].copy()
        for ds in self[1:]:
            dataset.stack(ds)
        return dataset

    def info_table(self, cumulative=False):
        """Get info table for datasets.

        Parameters
        ----------
        cumulative : bool
            Cumulate info across all observations

        Returns
        -------
        info_table : `~astropy.table.Table`
            Info table.
        """
        if not self.is_all_same_type:
            raise ValueError("Info table not supported for mixed dataset type.")

        stacked = self[0].copy()

        rows = [stacked.info_dict()]

        for dataset in self[1:]:
            if cumulative:
                stacked.stack(dataset)
                row = stacked.info_dict()
            else:
                row = dataset.info_dict()

            rows.append(row)

        return table_from_row_data(rows=rows)

    def __getitem__(self, val):
        if isinstance(val, (int, slice)):
            return self._datasets[val]
        elif isinstance(val, str):
            for idx, dataset in enumerate(self._datasets):
                if val == dataset.name:
                    return self._datasets[idx]
            raise IndexError(f"No dataset: {val!r}")
        else:
            raise TypeError(f"Invalid type: {type(val)!r}")

    def __len__(self):
        return len(self._datasets)

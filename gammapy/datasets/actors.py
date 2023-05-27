# Licensed under a 3-clause BSD style license - see LICENSE.rst
import inspect
import logging
import numpy as np
import ray
from gammapy.modeling.models import DatasetModels
from .core import Dataset, Datasets
from .map import MapDataset

log = logging.getLogger(__name__)


__all__ = ["DatasetsActor", "MapDatasetActor"]


class DatasetsActor(Datasets):
    """A modified Dataset collection for parallel evaluation using ray actors.
    Support only datasets composed of MapDataset.

    Parameters
    ----------
    datasets : `Datasets`
        Datasets
    """

    def __init__(self, datasets=None):
        if datasets is not None:
            actors = []
            datasets_list = []
            while datasets:
                d0 = datasets[0]
                datasets_list.append(MapDatasetActor(d0))
                datasets.remove(d0)  # moved to remote so removed from main process
            self._datasets = datasets_list
            self._actors = actors

    def insert(self, idx, dataset):
        if isinstance(dataset, Dataset):
            if dataset.name in self.names:
                raise (ValueError("Dataset names must be unique"))
            self._datasets.insert(idx, MapDatasetActor(dataset))
            self._actors.insert(
                idx,
            )
        else:
            raise TypeError(f"Invalid type: {type(dataset)!r}")

    def __getattr__(self, attr):
        """get attribute from remote each dataset"""

        def wrapper(update_remote=False, **kwargs):
            if update_remote:
                self._update_remote_models()
            results = ray.get([d.actor.get_attr.remote(attr) for d in self._datasets])
            return [res(**kwargs) if inspect.ismethod(res) else res for res in results]

        return wrapper

    def _update_remote_models(self):
        args = [list(d.models) for d in self._datasets]
        ray.get(
            [d.actor.set_models.remote(arg) for d, arg in zip(self._datasets, args)]
        )

    def stat_sum(self):
        """Compute joint likelihood"""
        args = [d.models.parameters.get_parameter_values() for d in self._datasets]
        results = ray.get(
            [
                d.actor._update_stat_sum.remote(arg)
                for d, arg in zip(self._datasets, args)
            ]
        )
        return np.sum(results)


class MapDatasetActor(MapDataset):
    """MapDataset for parallel evaluation as a ray actor.

    Parameters
    ----------
    dataset : `MapDataset`
        MapDataset
    """

    def __init__(self, dataset):
        from ray import remote

        empty = MapDataset(name=dataset.name, models=dataset.models)
        self.__dict__.update(empty.__dict__)
        self.actor = remote(_MapDatasetActorBackend).remote(dataset)

    def _update_remote_models(self):
        ray.get(self.actor.set_models.remote(self.models))

    def get(self, attr, update_remote=False, **kwargs):
        """get attribute from remote dataset"""
        if update_remote:
            self._update_remote_models()
        result = ray.get(self.actor.get_attr.remote(attr))
        return result(**kwargs) if inspect.ismethod(result) else result


class _MapDatasetActorBackend(MapDataset):
    """MapDataset backend for parallel evaluation as a ray actor.

    Parameters
    ----------
    dataset : `MapDataset`
        MapDataset
    """

    def __init__(self, dataset):
        self.__dict__.update(dataset.__dict__)
        if self.models is None:
            self.models = DatasetModels()

    def _update_stat_sum(self, values):
        self.models.parameters.set_parameter_values(values)
        return self.stat_sum()

    def get_attr(self, attr):
        return getattr(self, attr)

    def set_models(self, models):
        self.models = models

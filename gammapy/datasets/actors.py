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
    Fore now only available if composed only of MapDataset.

    Parameters
    ----------
    datasets : `Datasets`
        Datasets
    """

    def __init__(self, datasets=None):
        from .map import MapDataset

        if datasets is not None:
            actors = []
            datasets_list = []
            while datasets:
                d0 = datasets[0]
                actors.append(MapDatasetActor.remote(d0))
                datasets_list.append(MapDataset(name=d0.name, models=d0.models))
                datasets.remove(d0)
            self._datasets = datasets_list
            self._actors = actors

    def insert(self, idx, dataset):
        from .map import MapDataset, MapDatasetActor

        if isinstance(dataset, Dataset):
            if dataset.name in self.names:
                raise (ValueError("Dataset names must be unique"))
            self._datasets.insert(
                idx, MapDataset(name=dataset.name, models=dataset.models)
            )

            self._actors.insert(idx, MapDatasetActor.remote(dataset))
        else:
            raise TypeError(f"Invalid type: {type(dataset)!r}")

    def __getattr__(self, attr):
        """get attribute from remote each dataset"""

        def wrapper(update_remote=False, **kwargs):
            if update_remote:
                self._update_remote_models()
            results = ray.get([a.get_attr.remote(attr) for a in self._actors])
            return [res(**kwargs) if inspect.ismethod(res) else res for res in results]

        return wrapper

    def _update_remote_models(self):
        args = [list(d.models) for d in self._datasets]
        ray.get([a.set_models.remote(arg) for a, arg in zip(self._actors, args)])

    def stat_sum(self):
        """Compute joint likelihood"""
        args = [d.models.parameters.get_parameter_values() for d in self._datasets]
        ray.get(
            [a.set_parameter_values.remote(arg) for a, arg in zip(self._actors, args)]
        )
        # blocked until set_parameters_factors on actors complete
        res = ray.get([a.stat_sum.remote() for a in self._actors])
        return np.sum(res)


@ray.remote
class MapDatasetActor(MapDataset):
    """A modified MapDataset for parallel evaluation as a ray actor.

    Parameters
    ----------
    dataset : `MapDataset`
        MapDataset
    """

    def __init__(self, dataset):
        self.__dict__.update(dataset.__dict__)
        if self.models is None:
            self.models = DatasetModels()

    def set_parameter_values(self, values):
        self.models.parameters.set_parameter_values(values)

    def get_models(self):
        return list(self.models)

    def set_models(self, models):
        self.models = models

    def get_attr(self, attr):
        return getattr(self, attr)

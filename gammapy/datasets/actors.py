# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import inspect
import logging
import numpy as np
from gammapy.modeling.models import DatasetModels
from .core import Dataset, Datasets
from .map import MapDataset

log = logging.getLogger(__name__)


__all__ = ["DatasetsActor"]


class DatasetsActor(Datasets):
    """A modified Dataset collection for parallel evaluation using ray actors.
    Support only datasets composed of MapDataset.

    Parameters
    ----------
    datasets : `Datasets`
        Datasets
    """

    def __init__(self, datasets=None):
        from ray import get

        log.warning(
            "Gammapy support for parallelisation with ray is still a prototype and is not fully functional."
        )

        if datasets is not None:
            datasets_list = []
            while datasets:
                d0 = datasets[0]
                if d0.tag != "MapDataset":
                    raise TypeError(
                        f"For now datasets parallel evaluation is only supported for MapDataset, got {d0.tag} instead"
                    )
                datasets_list.append(MapDatasetActor(d0))
                datasets.remove(d0)  # moved to remote so removed from main process
            self._datasets = datasets_list
            self._ray_get = get

    def insert(self, idx, dataset):
        if isinstance(dataset, Dataset):
            if dataset.name in self.names:
                raise (ValueError("Dataset names must be unique"))
            self._datasets.insert(idx, MapDatasetActor(dataset))
        else:
            raise TypeError(f"Invalid type: {type(dataset)!r}")

    def __getattr__(self, name):
        """Get attribute from remote each dataset."""

        def wrapper(**kwargs):
            results = self._ray_get(
                [d._get_remote(name, **kwargs) for d in self._datasets]
            )
            if "plot" in name:
                results = [res(**kwargs) for res in results]
            for d in self._datasets:
                d._to_update = {}
            return results

        if inspect.ismethod(getattr(self._datasets[0], name)):
            return wrapper
        else:
            return wrapper()

    def stat_sum(self):
        """Compute joint likelihood."""
        results = self._ray_get([d._update_stat_sum() for d in self._datasets])
        return np.sum(results)


class RayFrontendMixin(object):
    """Ray mixin for a local class that interact with a remote instance."""

    # TODO: abstract class ?

    @property
    def _remote_attr(self):
        return [
            key
            for key in self._public_attr
            if key not in self._mutable_attr + self._local_attr
        ]

    def __getattr__(self, name, **kwargs):
        """Get attribute from remote."""
        if name in self._remote_attr:
            results = self._ray_get(self._get_remote(name, **kwargs))
            self._to_update = {}
            return results
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in self._remote_attr:
            raise AttributeError("can't set attribute")
        else:
            super().__setattr__(name, value)
            if name in self._mutable_attr:
                self._to_update[name] = value
                self._cache[name] = copy.deepcopy(value)

    def _get_remote(self, attr, **kwargs):
        return self._actor._get.remote(attr, self._to_update, **kwargs)


class RayBackendMixin:
    """Ray mixin for the remote class."""

    def _get(self, name, to_update={}, **kwargs):
        for key, value in to_update.items():
            setattr(self, key, value)
        res = getattr(self, name)
        if inspect.ismethod(res) and "plot" not in name:
            return res(**kwargs)
        else:
            return res


class MapDatasetActor(RayFrontendMixin):
    """MapDataset for parallel evaluation as a ray actor.

    Parameters
    ----------
    dataset : `MapDataset`
        MapDataset
    """

    _mutable_attr = ["models", "mask_fit"]
    _local_attr = ["name"]  # immutable small enough to keep and acces locally
    _public_attr = [key for key in dir(MapDataset) if not key.startswith("__")]

    def __init__(self, dataset):
        from ray import get, remote

        self._ray_get = get
        self._actor = remote(_MapDatasetActorBackend).remote()
        self._actor._from_dataset.remote(dataset)
        self._name = dataset.name
        self._to_update = {}
        self._cache = {}
        if dataset.models is None:
            self.models = DatasetModels()
        else:
            self.models = dataset.models
        self.mask_fit = dataset.mask_fit
        self._to_update = {}  # models and mask_fit are already ok from actor init

    @property
    def name(self):
        return self._name

    def _update_stat_sum(self):
        values = self.models.parameters.get_parameter_values()
        output = self._actor._update_stat_sum.remote(values, self._to_update)
        self._to_update = {}
        return output

    def _get_remote(self, attr, **kwargs):
        if ~np.all(
            self.models.parameters.value == self._cache["models"].parameters.value
        ):
            self._to_update["models"] = self.models
            self._cache["models"] = self.models.copy()
        return self._actor._get.remote(attr, self._to_update, **kwargs)


class _MapDatasetActorBackend(MapDataset, RayBackendMixin):
    """MapDataset backend for parallel evaluation as a ray actor.

    Parameters
    ----------
    dataset : `MapDataset`
        MapDataset
    """

    def _from_dataset(self, dataset):
        self.__dict__.update(dataset.__dict__)
        if self.models is None:
            self.models = DatasetModels()

    def _update_stat_sum(self, values, to_update={}):
        for key, value in to_update.items():
            setattr(self, key, value)
        self.models.parameters.set_parameter_values(values)
        return self.stat_sum()

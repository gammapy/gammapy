# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from gammapy.datasets import Datasets
from gammapy.datasets.actors import DatasetsActor
from gammapy.modeling import Fit
from gammapy.maps import TimeMapAxis

__all__ = ["ResolvedEstimator"]

log = logging.getLogger(__name__)


class ResolvedEstimator:
    def __init__(self, model, axis):
        self._category = "other"
        if isinstance(axis, TimeMapAxis) or "time" in axis.unit.physical_type:
            self._category = "time"
            if model.temporal_model:
                raise TypeError(
                    "Can't iterate a model with a temporal component over a time axis."
                )
        elif axis.is_energy_axis():
            self._category = "energy"
        self._axis = axis
        self._model = model
        self._fit = Fit()

    def run(self, datasets):
        if not isinstance(datasets, DatasetsActor):
            datasets = Datasets(datasets)

        valid_intervals = []
        fit_result = []
        index = 0
        for bin_min, bin_max in self._axis.iter_by_edges:
            if self._category == "time":
                if not isinstance(self._axis, TimeMapAxis):
                    bin_min = bin_min + self._axis.reference_time
                    bin_max = bin_max + self._axis.reference_time

                datasets_to_fit = datasets.select_time(
                    time_min=bin_min,
                    time_max=bin_max,
                )

            elif self._category == "energy":
                datasets_to_fit = datasets.slice_by_energy(
                    energy_min=bin_min, energy_max=bin_max
                )
            else:
                datasets_to_fit = datasets

            if len(datasets_to_fit) == 0:
                log.info(
                    "No Dataset for the"
                    + self._category
                    + f"interval {bin_min} to {bin_max}. Skipping interval."
                )
                continue

            result = self.fit_model_in_bin(datasets_to_fit, index)
            fit_result.append(result)
            valid_intervals.append([bin_min, bin_max])
            index += 1

        return valid_intervals, fit_result

    def fit_model_in_bin(self, datasets, index):
        model_in_bin = self._model.copy(name="Model_bin_" + str(index))
        datasets.models = model_in_bin
        result = self._fit.run(datasets)
        return result

import logging
import astropy.units as u
from astropy.table import Table
from gammapy.data import GTI
from gammapy.datasets import Datasets
from gammapy.datasets.actors import DatasetsActor
from gammapy.estimators import Estimator
from gammapy.maps import TimeMapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import Models, SkyModel, SpectralModel
from gammapy.utils.pbar import progress_bar

__all__ = ["TimeResolvedSpectroscopyEstimator"]

log = logging.getLogger(__name__)


class TimeResolvedSpectroscopyEstimator(Estimator):
    tag = "TimeResolvedSpectroscopyEstimator"

    def __init__(self, model, time_intervals=None, fit=None, atol="1e-6 s"):
        if isinstance(model, SpectralModel):
            model = SkyModel(model)
        self._model = model
        self.models = Models()
        self.time_intervals = time_intervals
        self.atol = u.Quantity(atol)
        self.fit_convergence = []
        self.fit_result = []

        if fit is None:
            fit = Fit()

        self.fit = fit

    def run(self, datasets, store_fit_result=False):
        if self.models:
            self.models = Models()
            self.fit_convergence = []

        if not isinstance(datasets, DatasetsActor):
            datasets = Datasets(datasets)

        if self.time_intervals is None:
            gti = datasets.gti
        else:
            gti = GTI.from_time_intervals(self.time_intervals)

        gti = gti.union(overlap_ok=False, merge_equal=False)

        valid_intervals = []
        index = 0
        for t_min, t_max in progress_bar(
            gti.time_intervals, desc="Time intervals selection"
        ):
            datasets_to_fit = datasets.select_time(
                time_min=t_min, time_max=t_max, atol=self.atol
            )

            if len(datasets_to_fit) == 0:
                log.info(
                    f"No Dataset for the time interval {t_min} to {t_max}. Skipping interval."
                )
                continue

            self.models.append(
                self.fit_model_in_bin(datasets_to_fit, index, store_fit_result)
            )
            valid_intervals.append([t_min, t_max])
            index += 1

        gti = GTI.from_time_intervals(valid_intervals)

        self._gti = gti
        self.axis = TimeMapAxis.from_gti(gti)

        return self.create_table()

    def create_table(self):
        col_names = []
        col_unit = []
        for par in self._model.parameters.free_parameters:
            col_names.append(par.name)
            col_names.append(par.name + "_err")
            unt = par.unit
            if unt is u.Unit():
                unt = ""
            col_unit.append(unt)
            col_unit.append(unt)

        t = Table(names=col_names, units=col_unit)

        for i in range(self.axis.nbin):
            col_data = []
            for name in self._model.parameters.free_parameters.names:
                col_data.append(self.models[i].parameters[name].value)
                col_data.append(self.models[i].parameters[name].error)
            t.add_row(col_data)

        t.add_columns(self._gti.table.columns, indexes=[0, 0])
        return t

    def fit_model_in_bin(self, datasets, index, store_fit_result):
        model_in_bin = self._model.copy(name="Model_bin_" + str(index))
        datasets.models = model_in_bin
        result = self.fit.run(datasets)
        self.fit_convergence.append(result.success)
        if store_fit_result:
            self.fit_result.append(result)
        return model_in_bin

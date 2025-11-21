# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from itertools import repeat
import numpy as np
import astropy.units as u
import gammapy.utils.parallel as parallel
from gammapy.data import GTI
from gammapy.datasets import Datasets
from gammapy.datasets.actors import DatasetsActor
from gammapy.maps import LabelMapAxis, Map, TimeMapAxis
from gammapy.modeling.models import Models
from gammapy.utils.pbar import progress_bar
from .core import FluxPoints
from .sed import FluxPointsEstimator

__all__ = ["LightCurveEstimator"]

log = logging.getLogger(__name__)


class LightCurveEstimator(FluxPointsEstimator):
    """Estimate light curve.

    The estimator will apply flux point estimation on the source model component to datasets
    in each of the provided time intervals.  The normalisation, ``norm``, is the only
    parameter of the source model left free to vary. Other model components
    can be left free to vary with the reoptimize option.

    If no time intervals are provided, the estimator will use the time intervals
    defined by the datasets GTIs.

    To be included in the estimation, the dataset must have their GTI fully
    overlapping a time interval.

    Time intervals without any dataset GTI fully overlapping will be dropped. They will not
    be stored in the final lightcurve `FluxPoints` object.

    Parameters
    ----------
    time_intervals : list of `~astropy.time.Time` objects
        Start and stop time for each interval to compute the LC.
    source : str or int, optional
        For which source in the model to compute the flux points.
        Default is 0, i.e. the first source of the models.
    atol : `~astropy.units.Quantity`, optional
        Tolerance value for time comparison with different scale. Default 1e-6 sec.
    n_sigma : float, optional
        Number of sigma to use for asymmetric error computation. Must be a positive value.
        Default is 1.
    n_sigma_ul : float, optional
        Number of sigma to use for upper limit computation. Must be a positive value.
        Default is 2.
    selection_optional : list of str, optional
        Which steps to execute. Available options are:

            * "all": all the optional steps are executed.
            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optional steps are not executed.
    energy_edges : list of `~astropy.units.Quantity`, optional
        Edges of the lightcurve energy bins. The resulting bin edges won't be exactly equal to the input ones,
        but rather the closest values to the energy axis edges of the parent dataset.
        Default is None: apply the estimator in each energy bin of the parent dataset.
        For further explanation see :ref:`estimators`.
    fit : `~gammapy.modeling.Fit`, optional
        Fit instance specifying the backend and fit options. If None, the `~gammapy.modeling.Fit` instance is created
        internally. Default is None.
    reoptimize : bool, optional
        If True the free parameters of the other models are fitted in each bin independently,
        together with the norm of the source of interest
        (but the other parameters of the source of interest are kept frozen).
        If False only the norm of the source of interest if fitted,
        and all other parameters are frozen at their current values.
        Default is False.
    stack_over_time_interval : bool, optional
        Whether to stack datasets within each time interval. Default is False.
        The predicted background counts from all datasets in the given interval
        will be stacked together, and the final background model (if any) will not
        have any free parameters. Available only if ``reoptimize`` is False.
    n_jobs : int, optional
        Number of processes used in parallel for the computation. Default is one,
        unless `~gammapy.utils.parallel.N_JOBS_DEFAULT` was modified. The number
        of jobs is limited to the number of physical CPUs.
    parallel_backend : {"multiprocessing", "ray"}, optional
        Which backend to use for multiprocessing. Defaults to `~gammapy.utils.parallel.BACKEND_DEFAULT`.
    norm : ~gammapy.modeling.Parameter` or dict, optional
        Norm parameter used for the fit.
        Default is None and a new parameter is created automatically,
        with value=1, name="norm", scan_min=0.2, scan_max=5, and scan_n_values = 11.
        By default, the min and max are not set and derived from the source model,
        unless the source model does not have one and only one norm parameter.
        If a dict is given the entries should be a subset of
        `~gammapy.modeling.Parameter` arguments.

    Examples
    --------
    For a usage example, see :doc:`/tutorials/analysis-time/light_curve` tutorial.

    Notes
    -----
    In case of failure of upper limits computation (e.g. nan), see the User Guide: :ref:`dropdown-UL`.
    """

    tag = "LightCurveEstimator"

    def __init__(
        self,
        time_intervals=None,
        atol="1e-6 s",
        stack_over_time_interval=False,
        **kwargs,
    ):
        self.time_intervals = time_intervals
        self.atol = u.Quantity(atol)
        self.stack_over_time_interval = stack_over_time_interval

        super().__init__(**kwargs)

    def run(self, datasets):
        """Run light curve extraction.

        Normalize integral and energy flux between emin and emax.

        Notes
        -----
        The progress bar can be displayed for this function.

        Parameters
        ----------
        datasets : list of `~gammapy.datasets.SpectrumDataset` or `~gammapy.datasets.MapDataset`
            Spectrum or Map datasets.

        Returns
        -------
        lightcurve : `~gammapy.estimators.FluxPoints`
            Light curve flux points.
        """
        if not isinstance(datasets, DatasetsActor):
            datasets = Datasets(datasets)

        if self.time_intervals is None:
            gti = datasets.gti
        else:
            gti = GTI.from_time_intervals(self.time_intervals)

        gti = gti.union(overlap_ok=False, merge_equal=False)

        rows = []
        valid_intervals = []
        parallel_datasets = []
        dataset_names = datasets.names
        for idx, (t_min, t_max) in enumerate(
            progress_bar(gti.time_intervals, desc="Time intervals selection")
        ):
            datasets_to_fit = datasets.select_time(
                time_min=t_min, time_max=t_max, atol=self.atol
            )

            if len(datasets_to_fit) == 0:
                log.info(
                    f"No Dataset for the time interval {t_min} to {t_max}. Skipping interval."
                )
                continue

            valid_intervals.append([t_min, t_max])
            if self.stack_over_time_interval and self.reoptimize:
                raise ValueError(
                    "It is not possible to stack the light curve if ``reoptimize=True``. You should set this value to False."
                )

            if self.stack_over_time_interval and not self.reoptimize:
                dataset_reduced = datasets_to_fit.stack_reduce(name="stacked")
                models = Models(datasets.models.copy())
                # Remove background models already applied in stack_reduce
                for model_name in models.background_models.values():
                    models.remove(model_name)
                for dataset in datasets:
                    models.reassign(dataset.name, dataset_reduced.name)
                dataset_reduced.models = models
                datasets_to_fit = Datasets([dataset_reduced])
                dataset_names = datasets_to_fit.names

            if self.n_jobs == 1:
                fp = self.estimate_time_bin_flux(datasets_to_fit, dataset_names)
                rows.append(fp)
            else:
                parallel_datasets.append(datasets_to_fit)

        if self.n_jobs > 1:
            self._update_child_jobs()
            rows = parallel.run_multiprocessing(
                self.estimate_time_bin_flux,
                zip(
                    parallel_datasets,
                    repeat(dataset_names),
                ),
                backend=self.parallel_backend,
                pool_kwargs=dict(processes=self.n_jobs),
                task_name="Time intervals",
            )

        if len(rows) == 0:
            raise ValueError("LightCurveEstimator: No datasets in time intervals")
        gti = GTI.from_time_intervals(valid_intervals)
        axis = TimeMapAxis.from_gti(gti=gti)
        return FluxPoints.from_stack(
            maps=rows,
            axis=axis,
        )

    @staticmethod
    def expand_map(m, dataset_names):
        """Expand map in dataset axis.

        Parameters
        ----------
        map : `~gammapy.maps.Map`
            Map to expand.
        dataset_names : list of str
            Dataset names.

        Returns
        -------
        map : `~gammapy.maps.Map`
            Expanded map.
        """
        label_axis = LabelMapAxis(labels=dataset_names, name="dataset")
        geom = m.geom.replace_axis(axis=label_axis)
        result = Map.from_geom(geom, data=np.nan)

        coords = m.geom.get_coord(sparse=True)
        result.set_by_coord(coords, vals=m.data)
        return result

    def estimate_time_bin_flux(self, datasets, dataset_names=None):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            List of dataset objects.
        dataset_names : list of str
            Dataset names.

        Returns
        -------
        result : `FluxPoints`
            Resulting flux points.
        """
        estimator = self.copy()
        estimator.n_jobs = self._n_child_jobs
        fp = estimator._run_flux_points(datasets)

        if dataset_names:
            for name in ["counts", "npred", "npred_excess"]:
                fp._data[name] = self.expand_map(
                    fp._data[name], dataset_names=dataset_names
                )
        return fp

    def _run_flux_points(self, datasets):
        return super().run(datasets)

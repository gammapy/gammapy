# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from itertools import repeat
import numpy as np
from astropy import units as u
from astropy.table import Table
import gammapy.utils.parallel as parallel
from gammapy.datasets import Datasets
from gammapy.datasets.actors import DatasetsActor
from gammapy.datasets.flux_points import _get_reference_model
from gammapy.maps import MapAxis, Map, RegionGeom
from gammapy.modeling import Fit, Parameters
from gammapy.modeling.utils import _parse_datasets
from gammapy.modeling.models import (
    PiecewiseNormSpectralModel,
    Models,
    PowerLawNormSpectralModel,
    TemplateNPredModel,
    TemplateSpatialModel,
)
from gammapy.stats.fit_statistics import GaussianPriorPenalty
from multiprocessing import Pool

from ..core import Estimator
from ..flux import FluxEstimator
from .core import FluxPoints

from scipy.interpolate import interp1d


log = logging.getLogger(__name__)

__all__ = ["FluxPointsEstimator"]


class FluxPointsEstimator(FluxEstimator, parallel.ParallelMixin):
    """Flux points estimator.

    Estimates flux points for a given list of datasets, energies and spectral model.

    To estimate the flux point the amplitude of the reference spectral model is
    fitted within the energy range defined by the energy group. This is done for
    each group independently. The amplitude is re-normalized using the "norm" parameter,
    which specifies the deviation of the flux from the reference model in this
    energy group. See https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/binned_likelihoods/index.html
    for details.

    The method is also described in the `Fermi-LAT catalog paper <https://ui.adsabs.harvard.edu/abs/2015ApJS..218...23A>`__
    or the `H.E.S.S. Galactic Plane Survey paper <https://ui.adsabs.harvard.edu/abs/2018A%26A...612A...1H>`__

    Parameters
    ----------
    source : str or int
        For which source in the model to compute the flux points.
    n_sigma : float, optional
        Number of sigma to use for asymmetric error computation. Must be a positive value.
        Default is 1.
    n_sigma_ul : float, optional
        Number of sigma to use for upper limit computation. Must be a positive value.
        Default is 2.
    n_sigma_sensitivity : float, optional
        Sigma to use for sensitivity computation. Must be a positive value.
        Default is 5.
    selection_optional : list of str, optional
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed.
            * "errn-errp": estimate asymmetric errors on flux.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.
            * "sensitivity": estimate sensitivity for a given significance.

        Default is None so the optional steps are not executed.
    energy_edges : list of `~astropy.units.Quantity`, optional
        Edges of the flux points energy bins. The resulting bin edges won't be exactly equal to the input ones,
        but rather the closest values to the energy axis edges of the parent dataset.
        Default is [1, 10] TeV.
    fit : `~gammapy.modeling.Fit`, optional
        Fit instance specifying the backend and fit options. If None, the `~gammapy.modeling.Fit` instance is created
        internally. Default is None.
    reoptimize : bool, optional
        If True, the free parameters of the other models are fitted in each bin independently,
        together with the norm of the source of interest
        (but the other parameters of the source of interest are kept frozen).
        If False, only the norm of the source of interest is fitted,
        and all other parameters are frozen at their current values.
        Default is False.
    sum_over_energy_groups : bool, optional
        Whether to sum over the energy groups or fit the norm on the full energy grid. Default is None.
    n_jobs : int, optional
        Number of processes used in parallel for the computation. The number of jobs is limited to the number of
        physical CPUs. If None, defaults to `~gammapy.utils.parallel.N_JOBS_DEFAULT`.
        Default is None.
    parallel_backend : {"multiprocessing", "ray"}, optional
        Which backend to use for multiprocessing. If None, defaults to `~gammapy.utils.parallel.BACKEND_DEFAULT`.
    norm : `~gammapy.modeling.Parameter` or dict, optional
        Norm parameter used for the fit.
        Default is None and a new parameter is created automatically, with value=1, name="norm",
        scan_min=0.2, scan_max=5, and scan_n_values = 11. By default, the min and max are not set
        (consider setting them if errors or upper limits computation fails). If a dict is given,
        the entries should be a subset of `~gammapy.modeling.Parameter` arguments.

    Notes
    -----
    - For further explanation, see :ref:`estimators`.
    - In case of failure of upper limits computation (e.g. nan), see the User Guide: :ref:`dropdown-UL`.

    Examples
    --------
    .. testcode::

        from astropy import units as u
        from gammapy.datasets import SpectrumDatasetOnOff
        from gammapy.estimators import FluxPointsEstimator
        from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

        path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
        dataset = SpectrumDatasetOnOff.read(path + "pha_obs23523.fits")

        pwl = PowerLawSpectralModel(index=2.7, amplitude='3e-11  cm-2 s-1 TeV-1')

        dataset.models = SkyModel(spectral_model=pwl, name="crab")

        estimator = FluxPointsEstimator(
            source="crab",
            energy_edges=[0.1, 0.3, 1, 3, 10, 30, 100] * u.TeV,
        )

        fp = estimator.run(dataset)
        print(fp)

    .. testoutput::

        FluxPoints
        ----------

          geom                   : RegionGeom
          axes                   : ['lon', 'lat', 'energy']
          shape                  : (1, 1, 6)
          quantities             : ['norm', 'norm_err', 'ts', 'npred', 'npred_excess', 'stat', 'stat_null', 'counts', 'success']
          ref. model             : pl
          n_sigma                : 1
          n_sigma_ul             : 2
          sqrt_ts_threshold_ul   : 2
          sed type init          : likelihood
    """

    tag = "FluxPointsEstimator"

    def __init__(
        self,
        energy_edges=[1, 10] * u.TeV,
        sum_over_energy_groups=False,
        n_jobs=None,
        parallel_backend=None,
        **kwargs,
    ):
        self.energy_edges = energy_edges
        self.sum_over_energy_groups = sum_over_energy_groups
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        fit = Fit(confidence_opts={"backend": "scipy"})
        kwargs.setdefault("fit", fit)
        super().__init__(**kwargs)

    def run(self, datasets):
        """Run the flux point estimator for all energy groups.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.

        Returns
        -------
        flux_points : `FluxPoints`
            Estimated flux points.
        """
        if not isinstance(datasets, DatasetsActor):
            datasets = Datasets(datasets=datasets)

        if not datasets.energy_axes_are_aligned:
            raise ValueError("All datasets must have aligned energy axes.")

        telescopes = []
        for d in datasets:
            if d.meta_table is not None and "TELESCOP" in d.meta_table.colnames:
                telescopes.extend(list(d.meta_table["TELESCOP"].flatten()))
        if len(np.unique(telescopes)) > 1:
            raise ValueError(
                "All datasets must use the same value of the"
                " 'TELESCOP' meta keyword."
            )

        meta = {
            "n_sigma": self.n_sigma,
            "n_sigma_ul": self.n_sigma_ul,
            "sed_type_init": "likelihood",
        }

        rows = parallel.run_multiprocessing(
            self.estimate_flux_point,
            zip(
                repeat(datasets),
                self.energy_edges[:-1],
                self.energy_edges[1:],
            ),
            backend=self.parallel_backend,
            pool_kwargs=dict(processes=self.n_jobs),
            task_name="Energy bins",
        )

        table = Table(rows, meta=meta)
        model = _get_reference_model(datasets.models[self.source], self.energy_edges)
        return FluxPoints.from_table(
            table=table,
            reference_model=model.copy(),
            gti=datasets.gti,
            format="gadf-sed",
        )

    def estimate_flux_point(self, datasets, energy_min, energy_max):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds to compute the flux point for.

        Returns
        -------
        result : dict
            Dictionary with results for the flux point.
        """
        datasets_sliced = datasets.slice_by_energy(
            energy_min=energy_min, energy_max=energy_max
        )
        if self.sum_over_energy_groups:
            datasets_sliced = datasets_sliced.__class__(
                [_.to_image(name=_.name) for _ in datasets_sliced]
            )

        if len(datasets_sliced) > 0:
            if datasets.models is not None:
                models_sliced = datasets.models._slice_by_energy(
                    energy_min=energy_min,
                    energy_max=energy_max,
                    sum_over_energy_groups=self.sum_over_energy_groups,
                )
                datasets_sliced.models = models_sliced

            return super().run(datasets=datasets_sliced)
        else:
            log.warning(f"No dataset contribute in range {energy_min}-{energy_max}")
            model = _get_reference_model(
                datasets.models[self.source], self.energy_edges
            )
            return self._nan_result(datasets, model, energy_min, energy_max)

    def _nan_result(self, datasets, model, energy_min, energy_max):
        """NaN result."""
        energy_axis = MapAxis.from_energy_edges([energy_min, energy_max])

        with np.errstate(invalid="ignore", divide="ignore"):
            result = model.reference_fluxes(energy_axis=energy_axis)
            # convert to scalar values
            result = {key: value.item() for key, value in result.items()}

        result.update(
            {
                "norm": np.nan,
                "stat": np.nan,
                "success": False,
                "norm_err": np.nan,
                "ts": np.nan,
                "counts": np.zeros(len(datasets)),
                "npred": np.nan * np.zeros(len(datasets)),
                "npred_excess": np.nan * np.zeros(len(datasets)),
                "datasets": datasets.names,
            }
        )

        if "errn-errp" in self.selection_optional:
            result.update({"norm_errp": np.nan, "norm_errn": np.nan})

        if "ul" in self.selection_optional:
            result.update({"norm_ul": np.nan})

        if "scan" in self.selection_optional:
            norm_scan = self.norm.copy().scan_values
            result.update({"norm_scan": norm_scan, "stat_scan": np.nan * norm_scan})

        if "sensitivity" in self.selection_optional:
            result.update({"norm_sensitivity": np.nan})

        return result


class RegularizedFluxPointsEstimator(Estimator):
    tag = "RegularizedFluxPointsEstimator"

    def __init__(
        self,
        energy_nodes,
        models,
        penalty_name="L2",
        lambda_=1,
        selection_optional=None,
        fit=None,
        reoptimize=False,
    ):
        self.energy_nodes = energy_nodes
        self.models = models
        self.penalty_name = penalty_name
        self.lambda_ = lambda_
        if self.penalty_name is None:
            self.penalty_name = "unpenalized"

        if fit is None:
            fit = Fit()
            self.fit = fit

        self.reoptimize = reoptimize
        self.selection_optional = selection_optional

    def _create_regularized_models(self, datasets):
        norm_models = Models()
        penalties = []
        for m in self.models:
            norm_model = m.copy(f"{m.name}_flux_points_{self.penalty_name}")
            norm_model.freeze()
            norm_model.spectral_model *= PiecewiseNormSpectralModel(
                energy=self.energy_nodes
            )

            if self.penalty_name == "unpenalized":
                penalty = None
            elif self.penalty_name == "L2":
                penalty = GaussianPriorPenalty.L2_penalty(
                    norm_model.parameters.free_parameters, mean=1, lambda_=self.lambda_
                )
            elif self.penalty_name == "smoothness":
                penalty = GaussianPriorPenalty.SmoothnessPenalty(
                    norm_model.parameters.free_parameters, mean=1, lambda_=self.lambda_
                )
            else:
                raise NotImplementedError

            if not self.penalty_name == "unpenalized":
                penalties.append(penalty)
            norm_models.append(norm_model)

        norm_models.set_penalties(penalties)
        self.norm_models_names = norm_models.names
        return norm_models

    def _create_region(self):
        e_axis = MapAxis.from_nodes(self.energy_nodes, name="energy", interp="log")
        return RegionGeom(region=None, axes=[e_axis])

    def compute_flux(self, datasets):
        Fit().run(datasets)

        map_dict = dict()
        for km, m in enumerate(self.models):
            name = self.norm_models_names[km]
            parameters = datasets.models[name].parameters.free_parameters

            geom = self._create_region()
            norm_map = Map.from_geom(geom, data=parameters.value, unit="")
            norm_err_map = Map.from_geom(
                geom, data=np.array([_.error for _ in parameters]), unit=""
            )
            stat_ref = datasets.stat_sum()
            ts_map = Map.from_geom(
                geom,
                data=np.array(
                    [self._compute_ts_param(datasets, stat_ref, p) for p in parameters]
                ),
                unit="",
            )

            map_dict[m.name] = {
                "norm": norm_map,
                "norm_err": norm_err_map,
                "ts": ts_map,
            }

        return map_dict

    @staticmethod
    def _compute_ts_param(datasets, stat_ref, parameter):
        with Parameters([parameter]).restore_status():
            parameter.value = 0
            stat_null = datasets.stat_sum()
        return stat_null - stat_ref

    def compute_errn_errp(self, datasets):
        map_dict = dict()
        for km, m in enumerate(self.models):
            name = self.norm_models_names[km]
            parameters = datasets.models[name].parameters.free_parameters

            results = []
            for par in parameters:
                results.append(self.fit.confidence(datasets, par))
            errn, errp = [], []
            for result in results:
                errn.append(result["errn"])
                errp.append(result["errp"])

            geom = self._create_region()
            norm_errn_map = Map.from_geom(geom, data=np.array(errn), unit="")
            norm_errp_map = Map.from_geom(geom, data=np.array(errp), unit="")
            map_dict[m.name] = {"norm_errn": norm_errn_map, "norm_errp": norm_errp_map}
        return map_dict

    def compute_ul(self, datasets):
        map_dict = dict()
        for km, m in enumerate(self.models):
            name = self.norm_models_names[km]
            parameters = datasets.models[name].parameters.free_parameters

            results = []
            bests = []
            for par in parameters:
                bests.append(par.value)
                results.append(self.fit.confidence(datasets, par, sigma=3))
            errp = []
            for result, best in zip(results, bests):
                errp.append(result["errp"] + best)

            geom = self._create_region()
            norm_ul_map = Map.from_geom(geom, data=np.array(errp), unit="")
            map_dict[m.name] = {"norm_ul": norm_ul_map}
        return map_dict

    def run(self, datasets):
        datasets, _ = _parse_datasets(datasets=datasets)

        datasets = datasets.copy()
        other_models = Models(
            [m for m in datasets.models if m.name not in self.models.names]
        )
        if not self.reoptimize:
            other_models.freeze()
        norm_models = self._create_regularized_models(datasets)
        models = other_models + norm_models
        models._penalties = (
            norm_models._penalties
        )  # TODO: should be supported by __add__
        print(models._penalties)

        datasets.models = models

        print(len(datasets.models.parameters.free_unique_parameters))
        print(datasets.models._penalties)

        maps_dict = self.compute_flux(datasets)
        print("stat_sum", datasets.stat_sum())
        print("stat_likelihood", datasets._stat_sum_likelihood())
        print("stat_prior", datasets.stat_sum() - datasets._stat_sum_likelihood())
        print(
            "stat_prior/stat_likelihood",
            (datasets.stat_sum() - datasets._stat_sum_likelihood())
            / datasets._stat_sum_likelihood(),
        )
        # print(maps_dict)
        # maps_dict = deep_update(maps_dict, self.compute_errn_errp(datasets))
        # print(maps_dict)

        # maps_dict = deep_update(maps_dict, self.compute_ul(datasets))
        # print(maps_dict)

        fp_dict = dict()
        for m in self.models:
            fp_dict[m.name] = FluxPoints.from_maps(
                maps=maps_dict[m.name],
                reference_model=m.spectral_model,
                sed_type="likelihood",
            )

        return dict(
            flux_points=fp_dict,
            models=models,
            stat_sum_likelihood=datasets._stat_sum_likelihood(),
            stat_sum_penalty=datasets.stat_sum() - datasets._stat_sum_likelihood(),
        )


class FluxPointsCollection:
    """Estimate the flux points from a collection of sources simultaneously.

    Parameters
    ----------
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the flux point bins.
    models : str or int
        Source models for which the flux points are computed (others are frozen).
    dataset : str or int
        Datasets used to compute the flus points.
        Datasets must share the same geometry.
    n_sigma_ul : int
        Number of sigma to use for upper limit computation. Default is 2.
    zero_min : int
        Forbid negative flux points or not (default id True).
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    """

    def __init__(
        self, energy_edges, models, datasets, n_sigma_ul=2, zero_min=True, fit=None
    ):
        for kd, d in enumerate(datasets):
            if kd == 0:
                self.geom = d.counts.geom
            elif d.counts.geom != self.geom:
                raise ValueError("Inconstistant geometries between datasets")
        self.n_sigma_ul = n_sigma_ul
        self.zero_min = zero_min

        if fit is None:
            fit = Fit()
            minuit_opts = {"tol": 0.1, "strategy": 1}
            fit.backend = "minuit"
            fit.optimize_opts = minuit_opts
        self.fit = fit

        self.models = models
        self.ns = len(models)

        # define energy edges
        self.energy_edges = energy_edges
        self.ne = len(self.energy_edges)
        self.e_centers = (energy_edges[:-1] * energy_edges[1:]) ** 0.5
        self.e_coords = self.geom.get_coord()["energy"]

        for d in datasets:
            d.npred()  # precompute npred
        self.datasets = datasets
        self.fp_datasets = self.prepare_datasets()

        print(
            "nfree :",
            len(self.fp_datasets.models.parameters.unique_parameters.free_parameters),
        )

        udnde = u.Unit("cm-2 s-1 TeV-1")
        self.fp_results = dict(
            npred=np.zeros((self.ne - 1, self.ns)),
            npred_err=np.zeros((self.ne - 1, self.ns)),
            dnde=np.zeros((self.ne - 1, self.ns)) * udnde,
            dnde_err=np.zeros((self.ne - 1, self.ns)) * udnde,
            dnde_ul=np.zeros((self.ne - 1, self.ns)) * udnde,
            ts=np.zeros((self.ne - 1, self.ns)),
        )

    def prepare_datasets(self):
        """define datasets with cached npred models to be renormalized"""

        self.spectral_models = {}
        for m in self.models:
            self.spectral_models[m.name] = PowerLawNormSpectralModel()
            if self.zero_min:
                self.spectral_models[m.name].norm.min = 0
            self.spectral_models[m.name].tilt.frozen = True

        fp_datasets = []
        for d in self.datasets:
            fp_dataset = d.copy(name=d.name)
            bkg_model = [d.background_model] if d.background_model else []
            fp_dataset.models = bkg_model
            fp_models = []
            npred_frozen = Map.from_geom(self.geom, dtype=float)
            for name, ev in d.evaluators.items():
                if ev.contributes:
                    npred = Map.from_geom(self.geom, dtype=float)
                    npred.stack(ev.compute_npred())
                    if name in Models(self.models).names:
                        fp_models.append(
                            TemplateNPredModel(
                                npred,
                                name=name + "_" + fp_dataset.name,
                                spectral_model=self.spectral_models[name],
                                datasets_names=[fp_dataset.name],
                            )
                        )
                    else:
                        npred_frozen.stack(npred)
                bkg_frozen = TemplateNPredModel(
                    npred_frozen,
                    name="frozen_" + fp_dataset.name,
                    datasets_names=[fp_dataset.name],
                )
            bkg_frozen.spectral_model.norm.frozen = True
            fp_dataset.models = Models(fp_models + [bkg_frozen] + bkg_model)
            fp_datasets.append(fp_dataset)
        return Datasets(fp_datasets)

    def get_mask(self, ke):
        mask_fit = Map.from_geom(self.geom, data=True)
        mask_fit.data &= (self.e_coords >= self.energy_edges[ke]) & (
            self.e_coords <= self.energy_edges[ke + 1 - self.ne]
        )
        return mask_fit

    def set_mask_fit(self, mask_fit):
        # redefine mask_fit
        for d in self.fp_datasets:
            d.mask_fit = mask_fit

    @staticmethod
    def compute_npred(datasets, param, model):
        "compute npred within the datasets masks"
        npred = 0
        for kd, d in enumerate(datasets):
            name = model.name + "_" + d.name
            if d.evaluators[name].contributes:
                npred_map = Map.from_geom(d.counts.geom)
                npred_map.stack(d.evaluators[name].compute_npred())
                npred += np.nansum(npred_map.data * d.mask.data) * param.value
            # apply mask for npred but only the slice mask_fit for dnde
        return npred

    @staticmethod
    def compute_dnde(energy, param, model, mask_fit):
        "compute differential flux"
        if isinstance(model.spatial_model, TemplateSpatialModel):
            spatial_integ = model.spatial_model.integrate_geom(mask_fit.geom)
            uspatial = spatial_integ.unit
            spatial_integ.data[~np.isfinite(spatial_integ.data)] = np.nan
            Ec = mask_fit.geom.axes[0].center
            spatial_integ = np.nansum(spatial_integ.data * mask_fit.data, axis=(1, 2))
            interp = interp1d(np.log10(Ec.value), spatial_integ, kind="linear")
            spatial_integ = interp(np.log10(energy.to(Ec.unit).value)) * uspatial

            spectral_values = model.spectral_model(energy)
            dnde = spectral_values * spatial_integ * param.value
        else:
            dnde = model.spectral_model(energy).squeeze() * param.value
        return dnde

    @staticmethod
    def compute_TS(datasets, param):
        """Test statistic against no source as null hypothesis"""
        cash = datasets.stat_sum()
        with Parameters([param]).restore_status():
            param.value = 0
            cash0 = datasets.stat_sum()
        return cash0 - cash

    def compute_flux_point(self, ke):
        """compute npred, dnde, TS, and ul (if necessasy)"""

        # redefine mask_fit
        mask_fit = self.get_mask(ke)
        self.set_mask_fit(mask_fit)

        # fit flux points
        self.fit.run(self.fp_datasets)

        udnde = u.Unit("cm-2 s-1 TeV-1")
        fp_results = dict(
            npred=np.zeros(self.ns),
            npred_err=np.zeros(self.ns),
            dnde=np.zeros(self.ns) * udnde,
            dnde_err=np.zeros(self.ns) * udnde,
            dnde_ul=np.zeros(self.ns) * udnde,
            ts=np.zeros(self.ns),
        )
        # compute quantities for each sources
        ks = 0
        for m, spec_corr in zip(self.models, self.spectral_models.values()):
            norm_param = spec_corr.norm
            norm = norm_param.value
            rel_err = norm_param.error / norm

            # compute npred
            npred = self.compute_npred(self.fp_datasets, norm_param, m)
            fp_results["npred"][ks] = npred
            fp_results["npred_err"][ks] = rel_err * npred

            # compute dnde
            dnde = self.compute_dnde(self.e_centers[ke], norm_param, m, mask_fit)
            fp_results["dnde"][ks] = dnde
            fp_results["dnde_err"][ks] = rel_err * dnde

            # compute TS
            TSnull = self.compute_TS(self.fp_datasets, norm_param)
            fp_results["ts"][ks] = TSnull

            # compute ul:
            if np.sign(TSnull) * np.sqrt(np.abs(TSnull)) < self.n_sigma_ul:
                res = self.fit.confidence(
                    datasets=self.fp_datasets,
                    parameter=norm_param,
                    sigma=self.n_sigma_ul,
                )
                fp_results["dnde_ul"][ks] = (1 + res["errp"] / norm_param.value) * dnde
            else:
                fp_results["dnde_ul"][ks] = np.nan * dnde
            ks += 1
        return fp_results

    def run(self, processes=1):
        """Compute flux point in each energy band in parallel over the given number of processes

        Parameters
        ----------
        processes : float
            Number of cores to use.
            Default is 1 so the evaluation is not done in parallel.

        Returns
        -------
        result : dict
            Dict with results for the flux point
        """
        if processes > 1:
            with Pool(processes=processes) as pool:
                res = pool.map(self.compute_flux_point, range(self.ne - 1))
        else:
            res = [self.compute_flux_point(ke) for ke in range(self.ne - 1)]
        for ke in range(self.ne - 1):
            fp_results = res[ke]
            self.fp_results["npred"][ke, :] = fp_results["npred"]
            self.fp_results["npred_err"][ke, :] = fp_results["npred_err"]
            self.fp_results["dnde"][ke, :] = fp_results["dnde"]
            self.fp_results["dnde_err"][ke, :] = fp_results["dnde_err"]
            self.fp_results["dnde_ul"][ke, :] = fp_results["dnde_ul"]
            self.fp_results["ts"][ke, :] = fp_results["ts"]
        return self.get_flux_points_dict()

    def get_flux_points_dict(self):
        """get flux points for each source

        Returns
        -------
        result : dict
            Dict of FluxPoints objects.
        """
        fp_dict = {}
        for km, m in enumerate(self.models):
            table = Table()
            table["e_min"] = self.energy_edges[:-1].to("TeV")
            table["e_max"] = self.energy_edges[1:].to("TeV")
            table["e_ref"] = self.e_centers.to("TeV")
            table["dnde"] = self.fp_results["dnde"][:, km]
            table["dnde_err"] = self.fp_results["dnde_err"][:, km]
            table["dnde_ul"] = self.fp_results["dnde_ul"][:, km]
            table["ts"] = self.fp_results["ts"][:, km]
            table.meta["SED_TYPE"] = "dnde"
            flux_points = FluxPoints.from_table(table, reference_model=m)
            fp_dict[m.name] = flux_points
        return fp_dict

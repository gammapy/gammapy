# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from itertools import repeat
import numpy as np
import scipy.stats as stats
from astropy import units as u
from astropy.table import Table
import gammapy.utils.parallel as parallel
from gammapy.datasets import Datasets
from gammapy.datasets.utils import set_and_restore_mask_fit
from gammapy.datasets.actors import DatasetsActor
from gammapy.datasets.flux_points import _get_reference_model
from gammapy.maps import MapAxis, Map
from gammapy.modeling import Fit, Sampler, Parameters, Parameter
from gammapy.modeling.models import (
    Models,
    PowerLawNormSpectralModel,
    TemplateNPredModel,
    UniformPrior,
)

from ..flux import FluxEstimator
from .core import FluxPoints

log = logging.getLogger(__name__)

__all__ = ["FluxPointsEstimator", "FluxCollectionEstimator"]


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
    allow_multiple_telescopes : bool, optional
        Whether to allow the computation for different telescopes.
        **WARNING**: This is currently an experimental feature.
        Default is False.

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
        allow_multiple_telescopes=False,
        **kwargs,
    ):
        self.energy_edges = energy_edges
        self.sum_over_energy_groups = sum_over_energy_groups
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.allow_multiple_telescopes = allow_multiple_telescopes

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
        if len(np.unique(telescopes)) > 1 and not self.allow_multiple_telescopes:
            raise ValueError(
                "All datasets must use the same value of the 'TELESCOP' meta keyword."
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


class FluxCollectionEstimator:
    """Estimate the flux points from a collection of sources simultaneously.

    This estimator computes spectral flux points for a *collection of sources*
    over a set of predefined energy bins. The normalizations of all
    ``SkyModel`` objects listed in ``models`` are fitted jointly in each energy bin,
    while all other ``SkyModel`` components in the datasets remain frozen.
    Re-optimization of each dataset’s background model is optional.

    The operation can be performed either with the standard likelihood optimizer
    (``~gammapy.modeling.Fit``) or with a sampler
    (``~gammapy.modeling.Sampler``), which can be used to derive asymmetric
    errors and upper limits.


    Parameters
    ----------
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the flux point bins.
    models : ~gammapy.modeling.Models` or list
        Source models for which the flux points are computed (others are frozen).
    n_sigma : float, optional
        Number of sigma to use for asymmetric error computation. Must be a positive value.
        Default is 1.
    n_sigma_ul : float, optional
        Number of sigma to use for upper limit computation. Must be a positive value.
        Default is 2.
    norm : `~gammapy.modeling.Parameter` or dict, optional
        Norm parameter used for the fit.
        Default is None and a new parameter is created with value=1, name="norm".
        If the `solver` is a sampler the default prior is uniform between [-10, 10].
    solver : `~gammapy.modeling.Fit` or `~gammapy.modeling.Sampler`
        Fit or Sampler instance specifying the backend and options.
        Default is a Sampler with options live_points=300, frac_remain=0.3.
    reoptimize : bool
        Whether to reoptimize each dataset.background_model. Default is False.
        Only SkyModel given in `models` will be fitted the others remain frozen,
        regardless of this option.
    selection_optional : list of str, optional
        Which additional quantities to estimate. Available options are:
            * "errn-errp": estimate asymmetric errors on flux.
        Fit solver computes upper limits if sqrt(TS) < n_sigma_ul.
        Sampler solver always compute errn-errp and ul.
    """

    def __init__(
        self,
        energy_edges,
        models,
        n_sigma=1,
        n_sigma_ul=2,
        norm=None,
        solver=None,
        reoptimize=False,
        selection_optional=None,
    ):
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul

        self.models = models
        self.ns = len(models)

        if solver is None:
            solver = Sampler(
                backend="ultranest",
                sampler_opts={"live_points": 300, "frac_remain": 0.3},
            )
        self.solver = solver
        self.reoptimize = reoptimize

        if selection_optional is None:
            selection_optional = []
        self.selection_optional = selection_optional

        if norm is None:
            norm = self._build_default_norm()
        self.norm = norm

        self.energy_edges = energy_edges

        self.energy_unit = "TeV"
        self.dnde_unit = u.Unit("cm-2 s-1 TeV-1")

    def _build_default_norm(self):
        prior = (
            UniformPrior(min=-10, max=10) if isinstance(self.solver, Sampler) else None
        )
        return Parameter(name="norm", value=1, unit="", prior=prior)

    @property
    def _available_keys(self):
        # TODO: what about 'npred' key
        keys = ["norm", "norm_ul", "ts"]
        if isinstance(self.solver, Fit):
            keys.append("norm_err")
        if isinstance(self.solver, Sampler) or "errn-errp" in self.selection_optional:
            keys.extend(["norm_errn", "norm_errp"])
        return keys

    @property
    def _energy_axis(self):
        """Energy axis for the input edges."""
        return MapAxis.from_energy_edges(self.energy_edges, name="energy", interp="log")

    def _prepare_datasets(self, datasets):
        """define datasets with cached npred models to be renormalized"""

        spectral_models = {}
        for m in self.models:
            spectral_models[m.name] = PowerLawNormSpectralModel(norm=self.norm.copy())
            spectral_models[m.name].tilt.frozen = True

        fp_datasets = []
        for d in datasets:
            fp_dataset = d.copy(name=d.name)
            bkg_model = self._get_bkg(d)
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
                                spectral_model=spectral_models[name],
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
            # keep this order such as fp_models parameters appear first in the samples indexing
            fp_datasets.append(fp_dataset)
        return Datasets(fp_datasets), spectral_models

    def _get_bkg(self, d):
        if d.background_model:
            bkg_model = [d.background_model.copy(name=d.background_model.name)]
            if not self.reoptimize:
                bkg_model[0].freeze()
        else:
            bkg_model = []
        return bkg_model

    @staticmethod
    def _compute_npred(datasets, param, model):
        "compute npred within the datasets masks"
        npred = 0
        for kd, d in enumerate(datasets):
            name = model.name + "_" + d.name
            if d.evaluators[name].contributes:
                npred_map = Map.from_geom(d.counts.geom)
                npred_map.stack(d.evaluators[name].compute_npred())
                npred += np.nansum(npred_map.data * d.mask.data) * param.value
        return npred

    @staticmethod
    def _compute_ts(datasets, param):
        """Test statistic against no source as null hypothesis"""
        cash = datasets._stat_sum_likelihood()
        with Parameters([param]).restore_status():
            param.value = 0
            cash0 = datasets._stat_sum_likelihood()
        return cash0 - cash

    def _run_fit(self, fp_datasets, spectral_models):
        """Compute flux estimates, uncertainties and UL using log-likelihood profile (i.e. using `~gammapy.modeling.Fit`)."""
        fit_results = self.solver.run(fp_datasets)
        # TODO: rename self.ns
        fp_result = {key: np.zeros(self.ns) for key in self._available_keys}
        fp_result["npred"] = np.zeros(self.ns)  # TODO: remove
        fp_result["solver_results"] = fit_results

        for km, (m, spec) in enumerate(zip(self.models, spectral_models.values())):
            norm_param = spec.norm
            norm = norm_param.value
            error = norm_param.error

            npred = self._compute_npred(fp_datasets, norm_param, m)
            fp_result["npred"][km] = npred

            fp_result["norm"][km] = norm

            fp_result["norm_err"][km] = error
            if "errn-errp" in self.selection_optional:
                res = self.solver.confidence(
                    datasets=fp_datasets,
                    parameter=norm_param,
                    sigma=self.n_sigma,
                )
                fp_result["norm_errn"][km] = res["errn"]
                fp_result["norm_errp"][km] = res["errp"]

            ts_null = self._compute_ts(fp_datasets, norm_param)
            fp_result["ts"][km] = ts_null

            if np.sign(ts_null) * np.sqrt(np.abs(ts_null)) < self.n_sigma_ul:
                res = self.solver.confidence(
                    datasets=fp_datasets,
                    parameter=norm_param,
                    sigma=self.n_sigma_ul,
                )
                fp_result["norm_ul"][km] = norm + res["errp"]
            else:
                fp_result["norm_ul"][km] = np.nan
        return fp_result

    def _run_sampler(self, fp_datasets, spectral_models):
        """compute npred, dnde, TS, errn, errp, and ul"""

        sampler_results = self.solver.run(fp_datasets).sampler_results

        points = sampler_results["weighted_samples"][
            "points"
        ]  # shape [n_samples, n_sources]
        weights = sampler_results["weighted_samples"]["weights"]

        cdf = stats.norm.cdf

        quantiles = {
            "norm": 50,
            "norm_errn": 100 * cdf(-self.n_sigma),
            "norm_errp": 100 * cdf(self.n_sigma),
            "norm_ul": 100 * cdf(self.n_sigma_ul),
        }

        # TODO: rename self.ns
        fp_result = {key: np.zeros(self.ns) for key in self._available_keys}
        fp_result["npred"] = np.zeros(self.ns)  # TODO: remove
        fp_result["solver_results"] = sampler_results

        for km, (m, spec) in enumerate(zip(self.models, spectral_models.values())):
            samples = points[:, km]
            norm_param = spec.norm

            percentiles = {
                k: np.percentile(samples, q, weights=weights, method="inverted_cdf")
                for k, q in quantiles.items()
            }

            norm = percentiles["norm"]
            norm_param.value = norm  # set before TS computation below

            fp_result["norm"][km] = norm
            fp_result["norm_errn"][km] = norm - percentiles["norm_errn"]
            fp_result["norm_errp"][km] = percentiles["norm_errp"] - norm
            fp_result["norm_ul"][km] = percentiles["norm_ul"]
            fp_result["npred"][km] = self._compute_npred(fp_datasets, norm_param, m)

        # compute TS after norm value is set to median for all models
        for km, spec in enumerate(spectral_models.values()):
            norm_param = spec.norm
            ts_null = self._compute_ts(fp_datasets, norm_param)
            fp_result["ts"][km] = ts_null

        return fp_result

    def run(self, datasets):
        """Compute flux point in each energy band

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets used to compute the flux points. They must share the same geometry.

        Returns
        -------
        result : dict
            Dict with results
        """
        datasets = Datasets(datasets)
        if len(datasets) == 0:
            raise ValueError("datasets cannot be empty")

        if not datasets.is_all_same_geom:
            raise ValueError("Inconsistent geometries between datasets")

        for d in datasets:
            d.npred()  # precompute npred

        self.geom = datasets[0]._geom
        fp_datasets, spectral_models = self._prepare_datasets(datasets)

        fp_results = []

        for ke, (emin, emax) in enumerate(self._energy_axis.iter_by_edges):
            with set_and_restore_mask_fit(
                fp_datasets, energy_min=emin, energy_max=emax
            ):
                if isinstance(self.solver, Sampler):
                    fp_result = self._run_sampler(fp_datasets, spectral_models)
                else:
                    fp_result = self._run_fit(fp_datasets, spectral_models)

            fp_results.append(fp_result)

        return self._get_flux_points_dict(fp_results)

    def _get_flux_points_dict(self, fp_results):
        """Extract flux points for each model from list of results.

        Parameters
        ----------
        fp_results : dict
            dict used to generate the flux points table.

        Returns
        -------
        result : dict
            Dict with results
        """

        def build_fp_from_idx(idx, model):
            table = Table()
            table["e_min"] = self._energy_axis.edges_min.to(self.energy_unit)
            table["e_max"] = self._energy_axis.edges_max.to(self.energy_unit)
            table["e_ref"] = self._energy_axis.center.to(self.energy_unit)
            table["ref_dnde"] = model(table["e_ref"]).to(self.dnde_unit)

            for key in self._available_keys:
                table[key] = np.array([fp[key][idx] for fp in fp_results])

            table.meta["SED_TYPE"] = "likelihood"

            return FluxPoints.from_table(
                table, reference_model=model.copy(), format="gadf-sed"
            )

        fp_dict = dict(
            energy_edges=self.energy_edges,
            solver_results=np.array(
                [fp["solver_results"] for fp in fp_results], dtype=object
            ),
            flux_points={},
        )

        for idx, m in enumerate(self.models):
            model = _get_reference_model(m, self.energy_edges)
            fp_dict["flux_points"][m.name] = build_fp_from_idx(idx, model)

        if isinstance(self.solver, Sampler):
            weights = [
                fp["solver_results"]["weighted_samples"]["weights"] for fp in fp_results
            ]
            dnde_dict = {}
            for model_idx, m in enumerate(self.models):
                dnde_dict[m.name] = []
                dnde_ref = fp_dict["flux_points"][m.name]["dnde_ref"].squeeze()
                for dnde, fp in zip(dnde_ref, fp_results):
                    points = fp["solver_results"]["weighted_samples"]["points"]
                    dnde_dict[m.name].append(dnde * points[:, model_idx])
            fp_dict["samples"] = dict(dnde=dnde_dict, weights=weights)

        return fp_dict

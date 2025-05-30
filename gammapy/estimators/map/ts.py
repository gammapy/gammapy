# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to compute test statistic images."""

import warnings
from itertools import repeat
import numpy as np
import scipy.optimize
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.coordinates import Angle
from astropy.utils import lazyproperty
import gammapy.utils.parallel as parallel
from gammapy.datasets import Datasets
from gammapy.datasets.map import MapEvaluator
from gammapy.datasets.utils import get_nearest_valid_exposure_position
from gammapy.maps import Map, MapAxis, Maps
from gammapy.modeling.models import PointSpatialModel, PowerLawSpectralModel, SkyModel
from gammapy.stats import cash, cash_sum_cython, f_cash_root_cython, norm_bounds_cython
from gammapy.stats.utils import ts_to_sigma
from gammapy.utils.array import shape_2N, symmetric_crop_pad_width
from gammapy.utils.pbar import progress_bar
from gammapy.utils.roots import find_roots
from ..core import Estimator
from ..utils import (
    _generate_scan_values,
    _get_default_norm,
    _get_norm_scan_values,
    estimate_exposure_reco_energy,
)
from .core import FluxMaps

__all__ = ["TSMapEstimator"]


def _extract_array(array, shape, position):
    """Helper function to extract parts of a larger array.

    Simple implementation of an array extract function , because
    `~astropy.ndata.utils.extract_array` introduces too much overhead.`

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array from which to extract.
    shape : tuple or int
        The shape of the extracted array.
    position : tuple of numbers or number
        The position of the small array's center with respect to the
        large array.
    """
    x_width = shape[2] // 2
    y_width = shape[1] // 2
    y_lo = position[0] - y_width
    y_hi = position[0] + y_width + 1
    x_lo = position[1] - x_width
    x_hi = position[1] + x_width + 1
    return array[:, y_lo:y_hi, x_lo:x_hi]


class TSMapEstimator(Estimator, parallel.ParallelMixin):
    r"""Compute test statistic map from a MapDataset using different optimization methods.

    The map is computed fitting by a single parameter norm fit. The fit is
    simplified by finding roots of the derivative of the fit statistics using
    various root finding algorithms. The approach is described in Appendix A
    in `Stewart (2009) <https://ui.adsabs.harvard.edu/abs/2009A%26A...495..989S/abstract>`_.

    The main output of this estimator is a `~gammapy.estimators.FluxMaps` object, which provides
    access to all computed quantities (see the example below and the `TSMapEstimator.run` function).

    Parameters
    ----------
    model : `~gammapy.modeling.models.SkyModel`
        Source model kernel. If set to None,
        assume spatail model: point source model, PointSpatialModel.
        spectral model: PowerLawSpectral Model of index 2
    kernel_width : `~astropy.coordinates.Angle`
        Width of the kernel to use: the kernel will be truncated at this size
    n_sigma : int
        Number of sigma for flux error. Default is 1.
    n_sigma_ul : int
        Number of sigma for flux upper limits. Default is 2.
    n_sigma_sensitivity : int
        Number of sigma for flux  sensitivity. Default is 5.
    downsampling_factor : int
        Sample down the input maps to speed up the computation. Only integer
        values that are a multiple of 2 are allowed. Note that the kernel is
        not sampled down, but must be provided with the downsampled bin size.
    threshold : float, optional
        If the test statistic value corresponding to the initial flux estimate is not above
        this threshold, the optimizing step is omitted to save computing time. Default is None.
    rtol : float
        Relative precision of the flux estimate. Used as a stopping criterion for
        the norm fit. Default is 0.01.
    selection_optional : list of str, optional
        Which maps to compute besides TS, sqrt(TS), flux and symmetric error on flux.
        Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric error on flux
            * "ul": estimate upper limits on flux.
            * "stat_scan": estimate likelihood profile
            * "sensitivity": estimate the flux sensitivity

        Default is None, so only "ts", "norm", "niter", "norm_err", "npred", "npred_excess", "stat", "stat_null" and
        "success" are computed.
    energy_edges : list of `~astropy.units.Quantity`, optional
        Edges of the target maps energy bins. The resulting bin edges won't be exactly equal to the input ones,
        but rather the closest values to the energy axis edges of the parent dataset.
        Default is None: apply the estimator in each energy bin of the parent dataset.
        For further explanation see :ref:`estimators`.
    sum_over_energy_groups : bool
        Whether to sum over the energy groups or fit the norm on the full energy
        cube.
    norm : `~gammapy.modeling.Parameter` or dict
        Norm parameter used for the likelihood profile computation on a fixed norm range.
        Only used for "stat_scan" in `selection_optional`.
        Default is None and a new parameter is created automatically,
        with value=1, name="norm", scan_min=-100, scan_max=100,
        and values sampled such as we can probe a 0.1% relative error on the norm.
        If a dict is given the entries should be a subset of
        `~gammapy.modeling.Parameter` arguments.
    n_jobs : int
        Number of processes used in parallel for the computation. Default is one,
        unless `~gammapy.utils.parallel.N_JOBS_DEFAULT` was modified. The number
        of jobs limited to the number of physical CPUs.
    parallel_backend : {"multiprocessing", "ray"}
        Which backend to use for multiprocessing. Defaults to `~gammapy.utils.parallel.BACKEND_DEFAULT`.
    max_niter : int
        Maximal number of iterations used by the root finding algorithm.
        Default is 100.

    Notes
    -----
    Negative :math:`TS` values are defined as following:

    .. math::
        TS = \left \{
                 \begin{array}{ll}
                   -TS \text{ if } F < 0 \\
                    TS \text{ else}
                 \end{array}
               \right.

    Where :math:`F` is the fitted flux norm.

    Examples
    --------
    >>> import astropy.units as u
    >>> from gammapy.estimators import TSMapEstimator
    >>> from gammapy.datasets import MapDataset
    >>> from gammapy.modeling.models import (SkyModel, PowerLawSpectralModel,PointSpatialModel)
    >>> spatial_model = PointSpatialModel()
    >>> spectral_model = PowerLawSpectralModel(amplitude="1e-22 cm-2 s-1 keV-1", index=2)
    >>> model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
    >>> dataset = MapDataset.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc.fits.gz")
    >>> estimator = TSMapEstimator(
    ...            model, kernel_width="1 deg", energy_edges=[10, 100] * u.GeV, downsampling_factor=4
    ...        )
    >>> maps = estimator.run(dataset)
    >>> print(maps)
    FluxMaps
    --------
    <BLANKLINE>
      geom                   : WcsGeom
      axes                   : ['lon', 'lat', 'energy']
      shape                  : (np.int64(400), np.int64(200), 1)
      quantities             : ['ts', 'norm', 'niter', 'norm_err', 'npred', 'npred_excess', 'stat', 'stat_null', 'success']
      ref. model             : pl
      n_sigma                : 1
      n_sigma_ul             : 2
      sqrt_ts_threshold_ul   : 2
      sed type init          : likelihood
    >>> print(maps.available_quantities))
    ['ts', 'norm', 'niter', 'norm_err', 'npred', 'npred_excess', 'stat', 'stat_null', 'success']
    """

    tag = "TSMapEstimator"
    _available_selection_optional = ["errn-errp", "ul", "stat_scan", "sensitivity"]

    def __init__(
        self,
        model=None,
        kernel_width=None,
        downsampling_factor=None,
        n_sigma=1,
        n_sigma_ul=2,
        n_sigma_sensitivity=5,
        threshold=None,
        rtol=0.01,
        selection_optional=None,
        energy_edges=None,
        sum_over_energy_groups=True,
        n_jobs=None,
        parallel_backend=None,
        norm=None,
        max_niter=100,
    ):
        if kernel_width is not None:
            kernel_width = Angle(kernel_width)

        self.kernel_width = kernel_width

        self.norm = _get_default_norm(norm, scan_values=_generate_scan_values())

        if model is None:
            model = SkyModel(
                spectral_model=PowerLawSpectralModel(),
                spatial_model=PointSpatialModel(),
                name="ts-kernel",
            )

        self.model = model
        self.downsampling_factor = downsampling_factor
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.n_sigma_sensitivity = n_sigma_sensitivity
        self.threshold = threshold
        self.rtol = rtol
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.sum_over_energy_groups = sum_over_energy_groups
        self.max_niter = max_niter

        self.selection_optional = selection_optional
        self.energy_edges = energy_edges
        self._flux_estimator = BrentqFluxEstimator(
            rtol=self.rtol,
            n_sigma=self.n_sigma,
            n_sigma_ul=self.n_sigma_ul,
            selection_optional=selection_optional,
            ts_threshold=threshold,
            norm=self.norm,
            max_niter=self.max_niter,
        )

    @property
    def selection_all(self):
        """Which quantities are computed."""
        selection = [
            "ts",
            "norm",
            "niter",
            "norm_err",
            "npred",
            "npred_excess",
            "stat",
            "stat_null",
            "success",
        ]

        if "stat_scan" in self.selection_optional:
            selection += [
                "dnde_scan_values",
                "stat_scan",
                "norm_errp",
                "norm_errn",
                "norm_ul",
            ]
        else:
            if "errn-errp" in self.selection_optional:
                selection += ["norm_errp", "norm_errn"]
            if "ul" in self.selection_optional:
                selection += ["norm_ul"]

        if "sensitivity" in self.selection_optional:
            selection += ["norm_sensitivity"]
        return selection

    def estimate_kernel(self, dataset):
        """Get the convolution kernel for the input dataset.

        Convolves the model with the IRFs at the center of the dataset,
        or at the nearest position with non-zero exposure.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.

        Returns
        -------
        kernel : `~gammapy.maps.Map`
            Kernel map.

        """
        geom = dataset.exposure.geom

        if self.kernel_width is not None:
            geom = geom.to_odd_npix(max_radius=self.kernel_width / 2)

        model = self.model.copy()
        model.spatial_model.position = geom.center_skydir

        # Creating exposure map with the mean non-null exposure
        exposure = Map.from_geom(geom, unit=dataset.exposure.unit)
        position = get_nearest_valid_exposure_position(
            dataset.exposure, geom.center_skydir
        )
        exposure_position = dataset.exposure.to_region_nd_map(position)
        if not np.any(exposure_position.data):
            raise ValueError(
                "No valid exposure. Impossible to compute kernel for TS Map."
            )
        exposure.data[...] = exposure_position.data

        # We use global evaluation mode to not modify the geometry
        evaluator = MapEvaluator(model=model)

        evaluator.update(
            exposure=exposure,
            psf=dataset.psf,
            edisp=dataset.edisp,
            geom=dataset.counts.geom,
            mask=dataset.mask_image,
        )

        kernel = evaluator.compute_npred()
        kernel.data /= kernel.data.sum()
        return kernel

    def estimate_flux_default(self, dataset, kernel=None, exposure=None):
        """Estimate default flux map using a given kernel.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.
        kernel : `~gammapy.maps.WcsNDMap`
            Source model kernel.
        exposure : `~gammapy.maps.WcsNDMap`
            Exposure map on reconstructed energy.

        Returns
        -------
        flux : `~gammapy.maps.WcsNDMap`
            Approximate flux map.
        """
        if exposure is None:
            exposure = estimate_exposure_reco_energy(dataset, self.model.spectral_model)

        if kernel is None:
            kernel = self.estimate_kernel(dataset=dataset)

        kernel = kernel.data / np.sum(kernel.data**2)

        with np.errstate(invalid="ignore", divide="ignore"):
            flux = (dataset.counts - dataset.npred()) / exposure
            flux.data = np.nan_to_num(flux.data)

        flux.quantity = flux.quantity.to("1 / (cm2 s)")
        flux = flux.convolve(kernel)
        if dataset.mask:
            flux *= dataset.mask
        return flux.sum_over_axes()

    @staticmethod
    def estimate_mask_default(dataset):
        """Compute default mask where to estimate test statistic values.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.

        Returns
        -------
        mask : `WcsNDMap`
            Mask map.
        """
        if dataset.mask_fit:
            mask = dataset.mask
        elif dataset.mask_safe:
            mask = dataset.mask_safe
        else:
            mask = Map.from_geom(dataset.counts.geom, data=True, dtype=bool)
        mask &= np.isfinite(dataset.npred())
        return mask

    def estimate_pad_width(self, dataset, kernel=None):
        """Estimate pad width of the dataset.

        Parameters
        ----------
        dataset : `MapDataset`
            Input MapDataset.
        kernel : `WcsNDMap`
            Source model kernel.

        Returns
        -------
        pad_width : tuple
            Padding width.
        """
        if kernel is None:
            kernel = self.estimate_kernel(dataset=dataset)

        geom = dataset.counts.geom.to_image()
        geom_kernel = kernel.geom.to_image()

        pad_width = np.array(geom_kernel.data_shape) // 2

        if self.downsampling_factor and self.downsampling_factor > 1:
            shape = tuple(np.array(geom.data_shape) + 2 * pad_width)
            pad_width = symmetric_crop_pad_width(geom.data_shape, shape_2N(shape))[0]

        return tuple(pad_width)

    def estimate_fit_input_maps(self, dataset):
        """Estimate fit input maps.

        Parameters
        ----------
        dataset : `MapDataset`
            Map dataset.

        Returns
        -------
        maps : dict of `Map`
            Maps dictionary.
        """
        # First create 2D map arrays

        exposure = estimate_exposure_reco_energy(dataset, self.model.spectral_model)

        kernel = self.estimate_kernel(dataset)

        mask = self.estimate_mask_default(dataset=dataset)

        flux = self.estimate_flux_default(
            dataset=dataset, kernel=kernel, exposure=exposure
        )

        counts = dataset.counts * mask
        background = dataset.npred() * mask
        exposure *= mask

        energy_axis = counts.geom.axes["energy"]

        flux_ref = self.model.spectral_model.integral(
            energy_axis.edges[0], energy_axis.edges[-1]
        )

        exposure_npred = (exposure * flux_ref).to_unit("")
        norm = (flux / flux_ref).to_unit("")

        if self.sum_over_energy_groups:
            weights = mask
            counts = counts.sum_over_axes()
            background = background.sum_over_axes()
            exposure_npred = exposure_npred.sum_over_axes()

        else:
            weights = None  # already applied

        return {
            "counts": counts,
            "background": background,
            "norm": norm,
            "mask": mask,
            "weights": weights,
            "exposure": exposure_npred,
            "kernel": kernel,
        }

    @staticmethod
    def _estimate_mask_2d(map_dict):
        """Compute default mask where to estimate test statistic values."""
        mask_2d = map_dict["mask"].reduce_over_axes(np.logical_or)
        mask_2d &= map_dict["background"].reduce_over_axes() > 0
        return mask_2d.data

    def estimate_flux_map(self, datasets):
        """Estimate flux and test statistic maps for single dataset.

        Parameters
        ----------
        dataset : `~gammapy.datasets.Datasets` or `~gammapy.datasets.MapDataset`
            Map dataset or Datasets (list of MapDataset with the same spatial geometry).
        """
        maps = [self.estimate_fit_input_maps(dataset=d) for d in datasets]

        # Combine all masks with an OR
        mask_2d = np.logical_or.reduce([self._estimate_mask_2d(_) for _ in maps])

        if not np.any(mask_2d):
            raise ValueError(
                """No valid positions found.
            Check that the dataset background is defined and not only zeros,
            or that the mask is not all False."
            """
            )

        x, y = np.where(np.squeeze(mask_2d))
        positions = list(zip(x, y))

        inputs = zip(
            positions,
            repeat([_["counts"].data.astype(float) for _ in maps]),
            repeat([_["exposure"].data.astype(float) for _ in maps]),
            repeat([_["background"].data.astype(float) for _ in maps]),
            repeat([_["kernel"].data for _ in maps]),
            repeat([_["norm"].data for _ in maps]),
            repeat([_["weights"] for _ in maps]),
            repeat(self._flux_estimator),
        )

        results = parallel.run_multiprocessing(
            _ts_value,
            inputs,
            backend=self.parallel_backend,
            pool_kwargs=dict(processes=self.n_jobs),
            task_name="TS map",
        )

        result = {}

        j, i = zip(*positions)

        geom = maps[0]["counts"].geom.squash(axis_name="energy")
        energy_axis = geom.axes["energy"]
        dnde_ref = self.model.spectral_model(energy_axis.center)

        for name in self.selection_all:
            if name in ["dnde_scan_values", "stat_scan"]:
                norm_bin_axis = MapAxis(
                    range(len(results[0]["dnde_scan_values"])),
                    interp="lin",
                    node_type="center",
                    name="dnde_bin",
                )

                axes = geom.axes + [norm_bin_axis]
                geom_scan = geom.to_image().to_cube(axes)

                if name == "dnde_scan_values":
                    unit = dnde_ref.unit
                    factor = dnde_ref.value
                else:
                    unit = ""
                    factor = 1

                m = Map.from_geom(geom_scan, data=np.nan, unit=unit)
                m.data[:, 0, j, i] = np.array([_[name] for _ in results]).T * factor

            else:
                m = Map.from_geom(geom=geom, data=np.nan, unit="")
                m.data[0, j, i] = [_[name] for _ in results]
            result[name] = m

        return result

    def run(self, datasets):
        """Run test statistic map estimation.

        Can compute predicted excess, TS, flux and error maps, and optionally upper limits or sensitivity.

        Requires a MapDataset with counts, exposure and background_model properly set to run.

        Notes
        -----
        The progress bar can be displayed for this function.

        Parameters
        ----------
        dataset : `~gammapy.datasets.Datasets` or `~gammapy.datasets.MapDataset`
            Map dataset or Datasets (list of MapDataset with the same spatial geometry).

        Returns
        -------
        maps : `~gammapy.estimators.FluxMaps`
            See the associated class for a complete description. In particular, it can contain the following maps:

                * npred_excess : predicted excess map
                * counts : counts map
                * ts : delta(TS) map
                * sqrt_ts : sqrt(delta(TS)), or significance map
                * flux : flux map
                * flux_err : symmetric error map
                * flux_ul : upper limit map.
                * flux_sensitivity : flux sensitivity for this dataset

        """
        datasets = Datasets(datasets)

        geom_ref = datasets[0].counts.geom
        for dataset in datasets:
            if dataset.stat_type != "cash":
                raise TypeError(
                    f"{type(dataset)} is not a valid type for {self.__class__}"
                )
            if dataset.counts.geom.to_image() != geom_ref.to_image():
                raise TypeError("Datasets geometries must match")

        datasets_models = datasets.models

        pad_width = (0, 0)
        for dataset in datasets:
            pad_width_dataset = self.estimate_pad_width(dataset=dataset)
            pad_width = tuple(np.maximum(pad_width, pad_width_dataset))

        datasets_padded = Datasets()
        for dataset in datasets:
            dataset = dataset.pad(pad_width, name=dataset.name)
            dataset = dataset.downsample(self.downsampling_factor, name=dataset.name)
            datasets_padded.append(dataset)
        datasets = datasets_padded

        energy_axis = self._get_energy_axis(dataset=datasets[0])

        results = []

        for energy_min, energy_max in progress_bar(
            energy_axis.iter_by_edges, desc="Energy bins"
        ):
            datasets_sliced = datasets.slice_by_energy(
                energy_min=energy_min, energy_max=energy_max
            )

            if datasets_models is not None:
                models_sliced = datasets_models._slice_by_energy(
                    energy_min=energy_min,
                    energy_max=energy_max,
                    sum_over_energy_groups=self.sum_over_energy_groups,
                )
                datasets_sliced.models = models_sliced
            result = self.estimate_flux_map(datasets_sliced)
            results.append(result)

        maps = Maps()

        for name in self.selection_all:
            m = Map.from_stack(maps=[_[name] for _ in results], axis_name="energy")

            order = 0 if name in ["niter", "success"] else 1
            m = m.upsample(
                factor=self.downsampling_factor, preserve_counts=False, order=order
            )

            maps[name] = m.crop(crop_width=pad_width)

        maps["success"].data = maps["success"].data.astype(bool)

        meta = {"n_sigma": self.n_sigma, "n_sigma_ul": self.n_sigma_ul}
        return FluxMaps(
            data=maps,
            reference_model=self.model,
            gti=dataset.gti,
            meta=meta,
        )


# TODO: merge with MapDataset?
class SimpleMapDataset:
    """Simple map dataset.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts array.
    background : `~numpy.ndarray`
        Background array.
    model : `~numpy.ndarray`
        Kernel array.

    """

    def __init__(self, model, counts, background, norm_guess):
        self.model = model
        self.counts = counts
        self.background = background
        self.norm_guess = norm_guess

    @lazyproperty
    def norm_bounds(self):
        """Bounds for x."""
        return norm_bounds_cython(self.counts, self.background, self.model)

    def npred(self, norm):
        """Predicted number of counts."""
        return self.background + norm * self.model

    def stat_sum(self, norm):
        """Statistics sum."""
        return cash_sum_cython(self.counts, self.npred(norm))

    def stat_sum_asimov(self, norm):
        """Statistics sum."""
        return cash_sum_cython(self.npred(norm), self.npred(norm))

    def stat_sum_asimov_null(self, norm):
        """Statistics sum."""
        return cash_sum_cython(self.npred(norm), self.background)

    def stat_derivative(self, norm):
        """Statistics derivative."""
        return f_cash_root_cython(norm, self.counts, self.background, self.model)

    def stat_2nd_derivative(self, norm):
        """Statistics 2nd derivative."""
        term_top = self.model**2 * self.counts
        term_bottom = (self.background + norm * self.model) ** 2
        mask = term_bottom == 0
        return (term_top / term_bottom)[~mask].sum()

    @classmethod
    def from_arrays(cls, counts, background, exposure, norm, position, kernel, weights):
        """"""
        if weights:
            # compute mask weighted kernel for the sum_over_axes case
            weights = _extract_array(weights.data, kernel.shape, position)
            kernel = (kernel * weights).sum(axis=0, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                kernel /= weights.sum(axis=0, keepdims=True)
                kernel[~np.isfinite(kernel)] = 0

        counts_cutout = _extract_array(counts, kernel.shape, position)
        background_cutout = _extract_array(background, kernel.shape, position)
        exposure_cutout = _extract_array(exposure, kernel.shape, position)
        model = kernel * exposure_cutout
        norm_guess = norm[0, position[0], position[1]]
        mask_invalid = (counts_cutout == 0) & (background_cutout == 0) & (model == 0)
        return cls(
            counts=counts_cutout[~mask_invalid],
            background=background_cutout[~mask_invalid],
            model=model[~mask_invalid],
            norm_guess=norm_guess,
        )


# TODO: merge with `FluxEstimator`?
class BrentqFluxEstimator(Estimator):
    """Single parameter flux estimator."""

    _available_selection_optional = ["errn-errp", "ul", "stat_scan", "sensitivity"]
    tag = "BrentqFluxEstimator"

    def __init__(
        self,
        rtol,
        n_sigma,
        n_sigma_ul,
        n_sigma_sensitivity=5,
        selection_optional=None,
        max_niter=100,
        ts_threshold=None,
        norm=None,
    ):
        self.rtol = rtol
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.n_sigma_sensitivity = n_sigma_sensitivity
        self.selection_optional = selection_optional
        self.max_niter = max_niter
        self.ts_threshold = ts_threshold
        self.norm = norm

    def estimate_best_fit(self, dataset):
        """Estimate best fit norm parameter.

        Parameters
        ----------
        dataset : `SimpleMapDataset`
            Simple map dataset.

        Returns
        -------
        result : dict
            Result dictionary including 'norm' and 'norm_err'.
        """
        # Compute norm bounds and assert counts > 0
        norm_min, norm_max, norm_min_total = dataset.norm_bounds

        if not dataset.counts.sum() > 0:
            norm, niter, success = norm_min_total, 0, True

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # here we do not use avoid find_roots for performance
                    result_fit = scipy.optimize.brentq(
                        f=dataset.stat_derivative,
                        a=norm_min,
                        b=norm_max,
                        maxiter=self.max_niter,
                        full_output=True,
                        rtol=self.rtol,
                    )
                    norm = max(result_fit[0], norm_min_total)
                    niter = result_fit[1].iterations
                    success = result_fit[1].converged
                except (RuntimeError, ValueError):
                    norm, niter, success = norm_min_total, self.max_niter, False

        with np.errstate(invalid="ignore", divide="ignore"):
            norm_err = np.sqrt(1 / dataset.stat_2nd_derivative(norm)) * self.n_sigma

        stat = dataset.stat_sum(norm=norm)
        stat_null = dataset.stat_sum(norm=0)

        return {
            "norm": norm,
            "norm_err": norm_err,
            "niter": niter,
            "ts": stat_null - stat,
            "stat": stat,
            "stat_null": stat_null,
            "success": success,
        }

    def _confidence(self, dataset, n_sigma, result, positive):
        stat_best = result["stat"]
        norm = result["norm"]
        norm_err = result["norm_err"]

        def ts_diff(x):
            return (stat_best + n_sigma**2) - dataset.stat_sum(x)

        if positive:
            min_norm = norm
            max_norm = norm + 1e2 * norm_err
            factor = 1
        else:
            min_norm = norm - 1e2 * norm_err
            max_norm = norm
            factor = -1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roots, res = find_roots(
                ts_diff,
                [min_norm],
                [max_norm],
                nbin=1,
                maxiter=self.max_niter,
                rtol=self.rtol,
            )
            # Where the root finding fails NaN is set as norm
            return (roots[0] - norm) * factor

    def estimate_ul(self, dataset, result):
        """Compute upper limit using likelihood profile method.

        Parameters
        ----------
        dataset : `SimpleMapDataset`
            Simple map dataset.

        Returns
        -------
        result : dict
            Result dictionary including 'norm_ul'.
        """
        flux_ul = result["norm"] + self._confidence(
            dataset=dataset, n_sigma=self.n_sigma_ul, result=result, positive=True
        )

        return {"norm_ul": flux_ul}

    def estimate_errn_errp(self, dataset, result):
        """Compute norm errors using likelihood profile method.

        Parameters
        ----------
        dataset : `SimpleMapDataset`
            Simple map dataset.

        Returns
        -------
        result : dict
            Result dictionary including 'norm_errp' and 'norm_errn'.
        """
        flux_errn = self._confidence(
            dataset=dataset, result=result, n_sigma=self.n_sigma, positive=False
        )
        flux_errp = self._confidence(
            dataset=dataset, result=result, n_sigma=self.n_sigma, positive=True
        )
        return {"norm_errn": flux_errn, "norm_errp": flux_errp}

    def estimate_sensitivity(self, dataset, result):
        norm = result["norm"]

        def sigma_diff(x):
            ts_asimov = dataset.stat_sum_asimov_null(x) - dataset.stat_sum_asimov(x)
            return (
                ts_to_sigma(ts_asimov, ts_asimov=ts_asimov) - self.n_sigma_sensitivity
            )

        min_norm = norm / 1000.0
        max_norm = norm * 1000.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roots, res = find_roots(
                sigma_diff,
                [min_norm],
                [max_norm],
                nbin=1,
                maxiter=self.max_niter,
                rtol=self.rtol,
            )
            # Where the root finding fails NaN is set as norm
            norm_sensitivity = roots[0]
        return {"norm_sensitivity": norm_sensitivity}

    def estimate_scan(self, dataset, result):
        """Compute likelihood profile.

        Parameters
        ----------
        dataset : `SimpleMapDataset`
            Simple map dataset.

        Returns
        -------
        result : dict
            Result dictionary including 'stat_scan'.
        """
        sparse_norms = _get_norm_scan_values(self.norm, result)

        scale = sparse_norms[None, :]
        model = dataset.model.ravel()[:, None]
        background = dataset.background.ravel()[:, None]
        counts = dataset.counts.ravel()[:, None]
        stat_scan = cash(counts, model * scale + background)

        stat_scan_local = stat_scan.sum(axis=0) - result["stat_null"]

        spline = InterpolatedUnivariateSpline(
            sparse_norms, stat_scan_local, k=1, ext="raise", check_finite=True
        )

        norms = np.unique(np.concatenate((sparse_norms, self.norm.scan_values)))
        stat_scan = spline(norms)

        ts = -stat_scan.min()
        ind = stat_scan.argmin()
        norm = norms[ind]

        maskp = norms > norm
        stat_diff = stat_scan - stat_scan.min()
        ind = np.abs(stat_diff - self.n_sigma**2)[~maskp].argmin()
        norm_errn = norm - norms[~maskp][ind]

        ind = np.abs(stat_diff - self.n_sigma**2)[maskp].argmin()
        norm_errp = norms[maskp][ind] - norm

        ind = np.abs(stat_diff - self.n_sigma_ul**2)[maskp].argmin()
        norm_ul = norms[maskp][ind]

        norm_err = (norm_errn + norm_errp) / 2

        return dict(
            ts=ts,
            norm=norm,
            norm_err=norm_err,
            norm_errn=norm_errn,
            norm_errp=norm_errp,
            norm_ul=norm_ul,
            stat_scan=stat_scan_local,
            dnde_scan_values=sparse_norms,
        )

    def estimate_default(self, dataset):
        """Estimate default norm.

        Parameters
        ----------
        dataset : `SimpleMapDataset`
            Simple map dataset.

        Returns
        -------
        result : dict
            Result dictionary including 'norm', 'norm_err' and "niter".
        """
        norm = dataset.norm_guess

        with np.errstate(invalid="ignore", divide="ignore"):
            norm_err = np.sqrt(1 / dataset.stat_2nd_derivative(norm)) * self.n_sigma

        stat = dataset.stat_sum(norm=norm)
        stat_null = dataset.stat_sum(norm=0)

        return {
            "norm": norm,
            "norm_err": norm_err,
            "niter": 0,
            "ts": stat_null - stat,
            "stat": stat,
            "stat_null": stat_null,
            "success": True,
        }

    def run(self, dataset):
        """Run flux estimator.

        Parameters
        ----------
        dataset : `SimpleMapDataset`
            Simple map dataset.

        Returns
        -------
        result : dict
            Result dictionary.
        """
        if self.ts_threshold is not None:
            result = self.estimate_default(dataset)
            if result["ts"] > self.ts_threshold:
                result = self.estimate_best_fit(dataset)
        else:
            result = self.estimate_best_fit(dataset)

        if "ul" in self.selection_optional:
            result.update(self.estimate_ul(dataset, result))

        if "errn-errp" in self.selection_optional:
            result.update(self.estimate_errn_errp(dataset, result))

        if "stat_scan" in self.selection_optional:
            result.update(self.estimate_scan(dataset, result))

        if "sensitivity" in self.selection_optional:
            result.update(self.estimate_sensitivity(dataset, result))

        norm = result["norm"]
        result["npred"] = dataset.npred(norm=norm).sum()
        result["npred_excess"] = result["npred"] - dataset.npred(norm=0).sum()
        result["stat"] = dataset.stat_sum(norm=norm)

        return result


def _ts_value(
    position, counts, exposure, background, kernel, norm, weights, flux_estimator
):
    """Compute test statistic value at a given pixel position.

    Uses approach described in Stewart (2009).

    Parameters
    ----------
    position : tuple (i, j)
        Pixel position.
    counts : `~numpy.ndarray`
        Counts image.
    background : `~numpy.ndarray`
        Background image.
    exposure : `~numpy.ndarray`
        Exposure image.
    kernel : `astropy.convolution.Kernel2D`
        Source model kernel.
    norm : `~numpy.ndarray`
        Norm image. The flux value at the given pixel position is used as
        starting value for the minimization.

    Returns
    -------
    TS : float
        Test statistic value at the given pixel position.
    """
    datasets = []
    nd = len(counts)
    for idx in range(nd):
        datasets.append(
            SimpleMapDataset.from_arrays(
                counts=counts[idx],
                background=background[idx],
                exposure=exposure[idx],
                norm=norm[idx],
                position=position,
                kernel=kernel[idx],
                weights=weights[idx],
            )
        )

    norm_guess = np.array([d.norm_guess for d in datasets])
    mask_valid = np.isfinite(norm_guess)
    if np.any(mask_valid):
        norm_guess = np.mean(norm_guess[mask_valid])
    else:
        norm_guess = 1.0
    dataset = SimpleMapDataset(
        counts=np.concatenate([d.counts for d in datasets]),
        background=np.concatenate([d.background for d in datasets]),
        model=np.concatenate([d.model for d in datasets]),
        norm_guess=norm_guess,
    )
    return flux_estimator.run(dataset)

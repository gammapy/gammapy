# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to compute TS images."""
import functools
import logging
import warnings
from multiprocessing import Pool
import numpy as np
import scipy.optimize
from astropy.coordinates import Angle
from astropy.utils import lazyproperty
from gammapy.datasets.map import MapEvaluator
from gammapy.datasets.utils import get_nearest_valid_exposure_position
from gammapy.maps import Map, Maps
from gammapy.modeling.models import PointSpatialModel, PowerLawSpectralModel, SkyModel
from gammapy.stats import cash_sum_cython, f_cash_root_cython, norm_bounds_cython
from gammapy.utils.array import shape_2N, symmetric_crop_pad_width
from gammapy.utils.pbar import progress_bar
from gammapy.utils.roots import find_roots
from ..core import Estimator
from ..utils import estimate_exposure_reco_energy
from .core import FluxMaps

__all__ = ["TSMapEstimator"]

log = logging.getLogger(__name__)


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


class TSMapEstimator(Estimator):
    r"""Compute TS map from a MapDataset using different optimization methods.

    The map is computed fitting by a single parameter norm fit. The fit is
    simplified by finding roots of the derivative of the fit statistics using
    various root finding algorithms. The approach is described in Appendix A
    in Stewart (2009).

    Parameters
    ----------
    model : `~gammapy.modeling.model.SkyModel`
        Source model kernel. If set to None,
        assume spatail model: point source model, PointSpatialModel.
        spectral model: PowerLawSpectral Model of index 2
    kernel_width : `~astropy.coordinates.Angle`
        Width of the kernel to use: the kernel will be truncated at this size
    n_sigma : int
        Number of sigma for flux error. Default is 1.
    n_sigma_ul : int
        Number of sigma for flux upper limits. Default is 2.
    downsampling_factor : int
        Sample down the input maps to speed up the computation. Only integer
        values that are a multiple of 2 are allowed. Note that the kernel is
        not sampled down, but must be provided with the downsampled bin size.
    threshold : float (None)
        If the TS value corresponding to the initial flux estimate is not above
        this threshold, the optimizing step is omitted to save computing time.
    rtol : float (0.01)
        Relative precision of the flux estimate. Used as a stopping criterion for
        the norm fit.
    selection_optional : list of str
        Which maps to compute besides TS, sqrt(TS), flux and symmetric error on flux.
        Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric error on flux.
            * "ul": estimate upper limits on flux.

        Default is None so the optional steps are not executed.
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the maps bins.
    sum_over_energy_groups : bool
        Whether to sum over the energy groups or fit the norm on the full energy
        cube.
    n_jobs : int
        Number of processes used in parallel for the computation.

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
                model, kernel_width="1 deg",energy_edges=[10, 100] * u.GeV, downsampling_factor=4
            )
    >>> maps = estimator.run(dataset)
    >>> print(maps)
    FluxMaps
    --------
    <BLANKLINE>
      geom                   : WcsGeom
      axes                   : ['lon', 'lat', 'energy']
      shape                  : (400, 200, 1)
      quantities             : ['ts', 'norm', 'niter', 'norm_err', 'npred', 'npred_excess', 'stat', 'stat_null', 'success']  # noqa: E501
      ref. model             : pl
      n_sigma                : 1
      n_sigma_ul             : 2
      sqrt_ts_threshold_ul   : 2
      sed type init          : likelihood


    References
    ----------
    [Stewart2009]_
    """
    tag = "TSMapEstimator"
    _available_selection_optional = ["errn-errp", "ul"]

    def __init__(
        self,
        model=None,
        kernel_width=None,
        downsampling_factor=None,
        n_sigma=1,
        n_sigma_ul=2,
        threshold=None,
        rtol=0.01,
        selection_optional=None,
        energy_edges=None,
        sum_over_energy_groups=True,
        n_jobs=None,
    ):
        if kernel_width is not None:
            kernel_width = Angle(kernel_width)

        self.kernel_width = kernel_width

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
        self.threshold = threshold
        self.rtol = rtol
        self.n_jobs = n_jobs
        self.sum_over_energy_groups = sum_over_energy_groups

        self.selection_optional = selection_optional
        self.energy_edges = energy_edges
        self._flux_estimator = BrentqFluxEstimator(
            rtol=self.rtol,
            n_sigma=self.n_sigma,
            n_sigma_ul=self.n_sigma_ul,
            selection_optional=selection_optional,
            ts_threshold=threshold,
        )

    @property
    def selection_all(self):
        """Which quantities are computed"""
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

        if "errn-errp" in self.selection_optional:
            selection += ["norm_errp", "norm_errn"]

        if "ul" in self.selection_optional:
            selection += ["norm_ul"]

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
        kernel : `Map`
            Kernel map

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
        return flux.sum_over_axes()

    @staticmethod
    def estimate_mask_default(dataset):
        """Compute default mask where to estimate TS values.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.

        Returns
        -------
        mask : `WcsNDMap`
            Mask map.
        """
        geom = dataset.counts.geom.to_image()

        mask = np.ones(geom.data_shape, dtype=bool)

        if dataset.mask is not None:
            mask &= dataset.mask.reduce_over_axes(func=np.logical_or, keepdims=False)

        # in some image there are pixels, which have exposure, but zero
        # background, which doesn't make sense and causes the TS computation
        # to fail, this is a temporary fix
        background = dataset.npred().sum_over_axes(keepdims=False)
        mask[background.data == 0] = False
        return Map.from_geom(data=mask, geom=geom)

    def estimate_pad_width(self, dataset, kernel=None):
        """Estimate pad width of the dataset

        Parameters
        ----------
        dataset : `MapDataset`
            Input MapDataset.
        kernel : `WcsNDMap`
            Source model kernel.

        Returns
        -------
        pad_width : tuple
            Padding width
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
        """Estimate fit input maps

        Parameters
        ----------
        dataset : `MapDataset`
            Map dataset

        Returns
        -------
        maps : dict of `Map`
            Maps dict
        """
        # First create 2D map arrays
        counts = dataset.counts
        background = dataset.npred()

        exposure = estimate_exposure_reco_energy(dataset, self.model.spectral_model)

        kernel = self.estimate_kernel(dataset)

        mask = self.estimate_mask_default(dataset=dataset)

        flux = self.estimate_flux_default(
            dataset=dataset, kernel=kernel, exposure=exposure
        )

        energy_axis = counts.geom.axes["energy"]

        flux_ref = self.model.spectral_model.integral(
            energy_axis.edges[0], energy_axis.edges[-1]
        )

        exposure_npred = (exposure * flux_ref * mask.data).to_unit("")

        norm = (flux / flux_ref).to_unit("")
        return {
            "counts": counts,
            "background": background,
            "norm": norm,
            "mask": mask,
            "exposure": exposure_npred,
            "kernel": kernel,
        }

    def estimate_flux_map(self, dataset):
        """Estimate flux and ts maps for single dataset

        Parameters
        ----------
        dataset : `MapDataset`
            Map dataset
        """
        maps = self.estimate_fit_input_maps(dataset=dataset)

        wrap = functools.partial(
            _ts_value,
            counts=maps["counts"].data.astype(float),
            exposure=maps["exposure"].data.astype(float),
            background=maps["background"].data.astype(float),
            kernel=maps["kernel"].data,
            norm=maps["norm"].data,
            flux_estimator=self._flux_estimator,
        )

        x, y = np.where(np.squeeze(maps["mask"].data))
        positions = list(zip(x, y))

        if self.n_jobs is None:
            results = list(map(wrap, positions))
        else:
            with Pool(processes=self.n_jobs) as pool:
                log.info("Using {} jobs to compute TS map.".format(self.n_jobs))
                results = pool.map(wrap, positions)

        result = {}

        j, i = zip(*positions)

        geom = maps["counts"].geom.squash(axis_name="energy")

        for name in self.selection_all:
            m = Map.from_geom(geom=geom, data=np.nan, unit="")
            m.data[0, j, i] = [_[name] for _ in results]
            result[name] = m

        return result

    def run(self, dataset):
        """
        Run TS map estimation.

        Requires a MapDataset with counts, exposure and background_model
        properly set to run.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input MapDataset.

        Returns
        -------
        maps : dict
             Dictionary containing result maps. Keys are:

                * ts : delta TS map
                * sqrt_ts : sqrt(delta TS), or significance map
                * flux : flux map
                * flux_err : symmetric error map
                * flux_ul : upper limit map

        """
        dataset_models = dataset.models

        pad_width = self.estimate_pad_width(dataset=dataset)
        dataset = dataset.pad(pad_width, name=dataset.name)
        dataset = dataset.downsample(self.downsampling_factor, name=dataset.name)

        energy_axis = self._get_energy_axis(dataset=dataset)

        results = []

        for energy_min, energy_max in progress_bar(
            energy_axis.iter_by_edges, desc="Energy bins"
        ):
            sliced_dataset = dataset.slice_by_energy(
                energy_min=energy_min, energy_max=energy_max, name=dataset.name
            )

            if self.sum_over_energy_groups:
                sliced_dataset = sliced_dataset.to_image(name=dataset.name)

            sliced_dataset.models = dataset_models
            result = self.estimate_flux_map(sliced_dataset)
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
    """Simple map dataset

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts array
    background : `~numpy.ndarray`
        Background array
    model : `~numpy.ndarray`
        Kernel array

    """

    def __init__(self, model, counts, background, norm_guess):
        self.model = model
        self.counts = counts
        self.background = background
        self.norm_guess = norm_guess

    @lazyproperty
    def norm_bounds(self):
        """Bounds for x"""
        return norm_bounds_cython(
            self.counts.ravel(), self.background.ravel(), self.model.ravel()
        )

    def npred(self, norm):
        """Predicted number of counts"""
        return self.background + norm * self.model

    def stat_sum(self, norm):
        """Stat sum"""
        return cash_sum_cython(self.counts.ravel(), self.npred(norm).ravel())

    def stat_derivative(self, norm):
        """Stat derivative"""
        return f_cash_root_cython(
            norm, self.counts.ravel(), self.background.ravel(), self.model.ravel()
        )

    def stat_2nd_derivative(self, norm):
        """Stat 2nd derivative"""
        term_top = self.model**2 * self.counts
        term_bottom = (self.background + norm * self.model) ** 2
        mask = term_bottom == 0
        return (term_top / term_bottom)[~mask].sum()

    @classmethod
    def from_arrays(cls, counts, background, exposure, norm, position, kernel):
        """"""
        counts_cutout = _extract_array(counts, kernel.shape, position)
        background_cutout = _extract_array(background, kernel.shape, position)
        exposure_cutout = _extract_array(exposure, kernel.shape, position)
        norm_guess = norm[0, position[0], position[1]]
        return cls(
            counts=counts_cutout,
            background=background_cutout,
            model=kernel * exposure_cutout,
            norm_guess=norm_guess,
        )


# TODO: merge with `FluxEstimator`?
class BrentqFluxEstimator(Estimator):
    """Single parameter flux estimator"""

    _available_selection_optional = ["errn-errp", "ul"]
    tag = "BrentqFluxEstimator"

    def __init__(
        self,
        rtol,
        n_sigma,
        n_sigma_ul,
        selection_optional=None,
        max_niter=20,
        ts_threshold=None,
    ):
        self.rtol = rtol
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.selection_optional = selection_optional
        self.max_niter = max_niter
        self.ts_threshold = ts_threshold

    def estimate_best_fit(self, dataset):
        """Estimate best fit norm parameter

        Parameters
        ----------
        dataset : `SimpleMapDataset`
            Simple map dataset

        Returns
        -------
        result : dict
            Result dict including 'norm' and 'norm_err'
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
            Simple map dataset

        Returns
        -------
        result : dict
            Result dict including 'norm_ul'
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
            Simple map dataset

        Returns
        -------
        result : dict
            Result dict including 'norm_errp' and 'norm_errn'
        """
        flux_errn = self._confidence(
            dataset=dataset, result=result, n_sigma=self.n_sigma, positive=False
        )
        flux_errp = self._confidence(
            dataset=dataset, result=result, n_sigma=self.n_sigma, positive=True
        )
        return {"norm_errn": flux_errn, "norm_errp": flux_errp}

    def estimate_default(self, dataset):
        """Estimate default norm

        Parameters
        ----------
        dataset : `SimpleMapDataset`
            Simple map dataset

        Returns
        -------
        result : dict
            Result dict including 'norm', 'norm_err' and "niter"
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
        """Run flux estimator

        Parameters
        ----------
        dataset : `SimpleMapDataset`
            Simple map dataset

        Returns
        -------
        result : dict
            Result dict
        """
        if self.ts_threshold is not None:
            result = self.estimate_default(dataset)
            if result["ts"] > self.ts_threshold:
                result = self.estimate_best_fit(dataset)
        else:
            result = self.estimate_best_fit(dataset)

        norm = result["norm"]
        result["npred"] = dataset.npred(norm=norm).sum()
        result["npred_excess"] = result["npred"] - dataset.npred(norm=0).sum()

        if "ul" in self.selection_optional:
            result.update(self.estimate_ul(dataset, result))

        if "errn-errp" in self.selection_optional:
            result.update(self.estimate_errn_errp(dataset, result))

        return result


def _ts_value(position, counts, exposure, background, kernel, norm, flux_estimator):
    """Compute TS value at a given pixel position.

    Uses approach described in Stewart (2009).

    Parameters
    ----------
    position : tuple (i, j)
        Pixel position.
    counts : `~numpy.ndarray`
        Counts image
    background : `~numpy.ndarray`
        Background image
    exposure : `~numpy.ndarray`
        Exposure image
    kernel : `astropy.convolution.Kernel2D`
        Source model kernel
    norm : `~numpy.ndarray`
        Norm image. The flux value at the given pixel position is used as
        starting value for the minimization.

    Returns
    -------
    TS : float
        TS value at the given pixel position.
    """
    dataset = SimpleMapDataset.from_arrays(
        counts=counts,
        background=background,
        exposure=exposure,
        kernel=kernel,
        position=position,
        norm=norm,
    )
    return flux_estimator.run(dataset)

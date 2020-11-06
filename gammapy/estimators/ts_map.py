# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to compute TS images."""
import contextlib
import functools
import logging
import warnings
from multiprocessing import Pool
import numpy as np
import scipy.optimize
from astropy import units as u
from astropy.coordinates import Angle
from astropy.utils import lazyproperty
from gammapy.datasets import Datasets
from gammapy.datasets.map import MapEvaluator
from gammapy.maps import Map, WcsGeom
from gammapy.modeling.models import PointSpatialModel, PowerLawSpectralModel, SkyModel
from gammapy.stats import cash_sum_cython, f_cash_root_cython, norm_bounds_cython
from gammapy.utils.array import shape_2N, symmetric_crop_pad_width
from .core import Estimator
from .utils import estimate_exposure_reco_energy

__all__ = ["TSMapEstimator"]

log = logging.getLogger(__name__)


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


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
    simplified by finding roots of the the derivative of the fit statistics using
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
    rtol : float (0.001)
        Relative precision of the flux estimate. Used as a stopping criterion for
        the norm fit.
    selection_optional : list of str or 'all'
        Which maps to compute besides TS, sqrt(TS), flux and symmetric error on flux.
        Available options are:

            * "errn-errp": estimate assymmetric error on flux.
            * "ul": estimate upper limits on flux.

        By default all steps are executed.
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

    References
    ----------
    [Stewart2009]_
    """
    tag = "TSMapEstimator"
    _available_selection_optional = ["errn-errp", "ul"]

    def __init__(
        self,
        model=None,
        kernel_width="0.2 deg",
        downsampling_factor=None,
        n_sigma=1,
        n_sigma_ul=2,
        threshold=None,
        rtol=0.01,
        selection_optional="all",
        energy_edges=None,
        sum_over_energy_groups=True,
        n_jobs=None,
    ):
        self.kernel_width = Angle(kernel_width)

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
        selection = ["ts", "flux", "niter", "flux_err"]

        if "errn-errp" in self.selection_optional:
            selection += ["flux_errp", "flux_errn"]

        if "ul" in self.selection_optional:
            selection += ["flux_ul"]

        return selection

    def estimate_kernel(self, dataset):
        """Get the convolution kernel for the input dataset.

        Convolves the model with the PSFKernel at the center of the dataset.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.

        Returns
        -------
        kernel : `Map`
            Kernel map

        """
        # TODO: further simplify the code below
        geom = dataset.counts.geom

        model = self.model.copy()
        model.spatial_model.position = geom.center_skydir

        binsz = np.mean(geom.pixel_scales)
        width_pix = self.kernel_width / binsz

        npix = round_up_to_odd(width_pix.to_value(""))

        axis = dataset.exposure.geom.axes["energy_true"]

        geom_kernel = WcsGeom.create(
            skydir=model.position, proj="TAN", npix=npix, axes=[axis], binsz=binsz
        )

        exposure = Map.from_geom(geom_kernel, unit="cm2 s1")
        exposure.data += 1.0

        # We use global evaluation mode to not modify the geometry
        evaluator = MapEvaluator(model, evaluation_mode="global")
        evaluator.update(exposure, dataset.psf, dataset.edisp, dataset.counts.geom)

        kernel = evaluator.compute_npred()
        kernel.data /= kernel.data.sum()

        if (self.kernel_width + binsz >= geom.width).any():
            raise ValueError(
                "Kernel shape larger than map shape, please adjust"
                " size of the kernel"
            )
        return kernel

    def estimate_flux_default(self, dataset, kernel, exposure=None):
        """Estimate default flux map using a given kernel.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.
        kernel : `~numpy.ndarray`
            Source model kernel.

        Returns
        -------
        flux : `~gammapy.maps.WcsNDMap`
            Approximate flux map.
        """
        if exposure is None:
            exposure = estimate_exposure_reco_energy(dataset, self.model.spectral_model)

        kernel = kernel / np.sum(kernel ** 2)
        flux = (dataset.counts - dataset.npred()) / exposure
        flux.quantity = flux.quantity.to("1 / (cm2 s)")
        flux = flux.convolve(kernel)
        return flux.sum_over_axes()

    @staticmethod
    def estimate_mask_default(dataset, kernel):
        """Compute default mask where to estimate TS values.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.
        kernel : `~numpy.ndarray`
            Source model kernel.

        Returns
        -------
        mask : `gammapy.maps.WcsNDMap`
            Mask map.
        """
        geom = dataset.counts.geom.to_image()

        # mask boundary
        mask = np.zeros(geom.data_shape, dtype=bool)
        slice_x = slice(kernel.shape[2] // 2, -kernel.shape[2] // 2 + 1)
        slice_y = slice(kernel.shape[1] // 2, -kernel.shape[1] // 2 + 1)
        mask[slice_y, slice_x] = 1

        if dataset.mask is not None:
            mask &= dataset.mask.reduce_over_axes(func=np.logical_or, keepdims=False)

        # in some image there are pixels, which have exposure, but zero
        # background, which doesn't make sense and causes the TS computation
        # to fail, this is a temporary fix
        background = dataset.npred().sum_over_axes(keepdims=False)
        mask[background.data == 0] = False
        return Map.from_geom(data=mask, geom=geom)

    def estimate_sqrt_ts(self, map_ts, norm):
        r"""Compute sqrt(TS) map.

        Compute sqrt(TS) as defined by:

        .. math::
            \sqrt{TS} = \left \{
            \begin{array}{ll}
              -\sqrt{-TS} & : \text{if} \ TS < 0 \\
              \sqrt{TS} & : \text{else}
            \end{array}
            \right.

        Parameters
        ----------
        map_ts : `gammapy.maps.WcsNDMap`
            Input TS map.

        Returns
        -------
        sqrt_ts : `gammapy.maps.WcsNDMap`
            Sqrt(TS) map.
        """
        sqrt_ts = self.get_sqrt_ts(map_ts.data, norm.data)
        return map_ts.copy(data=sqrt_ts)

    def estimate_flux_map(self, dataset):
        """Estimate flux and ts maps for single dataset

        Parameters
        ----------
        dataset : `MapDataset`
            Map dataset
        """
        # First create 2D map arrays
        counts = dataset.counts
        background = dataset.npred()

        exposure = estimate_exposure_reco_energy(dataset, self.model.spectral_model)

        kernel = self.estimate_kernel(dataset)

        mask = self.estimate_mask_default(dataset, kernel.data)

        flux = self.estimate_flux_default(dataset, kernel.data, exposure=exposure)

        energy_axis = counts.geom.axes["energy"]
        flux_ref = self.model.spectral_model.integral(
            energy_axis.edges[0], energy_axis.edges[-1]
        )
        exposure_npred = (exposure * flux_ref).quantity.to_value("")

        wrap = functools.partial(
            _ts_value,
            counts=counts.data.astype(float),
            exposure=exposure_npred.astype(float),
            background=background.data.astype(float),
            kernel=kernel.data,
            norm=(flux.quantity / flux_ref).to_value(""),
            flux_estimator=self._flux_estimator,
        )

        x, y = np.where(np.squeeze(mask.data))
        positions = list(zip(x, y))

        if self.n_jobs is None:
            results = list(map(wrap, positions))
        else:
            with contextlib.closing(Pool(processes=self.n_jobs)) as pool:
                log.info("Using {} jobs to compute TS map.".format(self.n_jobs))
                results = pool.map(wrap, positions)

            pool.join()

        result = {}

        j, i = zip(*positions)

        geom = counts.geom.squash(axis_name="energy")

        for name in self.selection_all:
            unit = 1 / exposure.unit if "flux" in name else ""
            m = Map.from_geom(geom=geom, data=np.nan, unit=unit)
            m.data[0, j, i] = [_[name.replace("flux", "norm")] for _ in results]
            if "flux" in name:
                m.data *= flux_ref.to_value(m.unit)
                m.quantity = m.quantity.to("1 / (cm2 s)")
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
        if self.downsampling_factor:
            shape = dataset.counts.geom.to_image().data_shape
            pad_width = symmetric_crop_pad_width(shape, shape_2N(shape))[0]
            dataset = dataset.pad(pad_width).downsample(self.downsampling_factor)

        # TODO: add support for joint likelihood fitting to TSMapEstimator
        datasets = Datasets(dataset)

        if self.energy_edges is None:
            energy_axis = dataset.counts.geom.axes["energy"]
            energy_edges = u.Quantity([energy_axis.edges[0], energy_axis.edges[-1]])
        else:
            energy_edges = self.energy_edges

        results = []

        for energy_min, energy_max in zip(energy_edges[:-1], energy_edges[1:]):
            sliced_dataset = datasets.slice_by_energy(energy_min, energy_max)[0]

            if self.sum_over_energy_groups:
                sliced_dataset = sliced_dataset.to_image()

            sliced_dataset.models = dataset_models
            result = self.estimate_flux_map(sliced_dataset)
            results.append(result)

        result_all = {}

        for name in self.selection_all:
            map_all = Map.from_images(images=[_[name] for _ in results])

            if self.downsampling_factor:
                order = 0 if name == "niter" else 1
                map_all = map_all.upsample(
                    factor=self.downsampling_factor, preserve_counts=False, order=order
                )
                map_all = map_all.crop(crop_width=pad_width)

            result_all[name] = map_all

        result_all["sqrt_ts"] = self.estimate_sqrt_ts(
            result_all["ts"], result_all["flux"]
        )
        return result_all


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
        with np.errstate(invalid="ignore", divide="ignore"):
            return (
                self.model ** 2
                * self.counts
                / (self.background + norm * self.model) ** 2
            ).sum()

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
        """Optimize for a single parameter"""
        result = {}
        # Compute norm bounds and assert counts > 0
        norm_min, norm_max, norm_min_total = dataset.norm_bounds

        if not dataset.counts.sum() > 0:
            norm, niter = norm_min_total, 0

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
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
                except (RuntimeError, ValueError):
                    # Where the root finding fails NaN is set as norm
                    norm, niter = norm_min_total, self.max_niter

        stat = dataset.stat_sum(norm=norm)
        stat_null = dataset.stat_sum(norm=0)
        result["ts"] = stat_null - stat
        result["norm"] = norm
        result["niter"] = niter

        with np.errstate(invalid="ignore", divide="ignore"):
            result["norm_err"] = (
                np.sqrt(1 / dataset.stat_2nd_derivative(norm)) * self.n_sigma
            )
        result["stat"] = stat
        return result

    def _confidence(self, dataset, n_sigma, result, positive):

        stat_best = result["stat"]
        norm = result["norm"]
        norm_err = result["norm_err"]

        def ts_diff(x):
            return (stat_best + n_sigma ** 2) - dataset.stat_sum(x)

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
            try:
                result_fit = scipy.optimize.brentq(
                    ts_diff, min_norm, max_norm, maxiter=self.max_niter, rtol=self.rtol,
                )
                return (result_fit - norm) * factor
            except (RuntimeError, ValueError):
                # Where the root finding fails NaN is set as norm
                return np.nan

    def estimate_ul(self, dataset, result):
        """"""

        flux_ul = self._confidence(
            dataset=dataset, n_sigma=self.n_sigma_ul, result=result, positive=True
        )

        return {"norm_ul": flux_ul}

    def estimate_errn_errp(self, dataset, result):
        """
        Compute norm errors using likelihood profile method.
        """

        flux_errn = self._confidence(
            dataset=dataset, result=result, n_sigma=self.n_sigma, positive=False
        )
        flux_errp = self._confidence(
            dataset=dataset, result=result, n_sigma=self.n_sigma, positive=True
        )
        return {"norm_errn": flux_errn, "norm_errp": flux_errp}

    def estimate_default(self, dataset):
        norm = dataset.norm_guess
        stat = dataset.stat_sum(norm=norm)
        stat_null = dataset.stat_sum(norm=0)
        ts = stat_null - stat

        with np.errstate(invalid="ignore", divide="ignore"):
            norm_err = np.sqrt(1 / dataset.stat_2nd_derivative(norm)) * self.n_sigma
        return {"norm": norm, "ts": ts, "norm_err": norm_err, "stat": stat, "niter": 0}

    def run(self, dataset):
        """"""
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

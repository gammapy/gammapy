# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to compute TS images."""
import functools
import logging
import warnings
import numpy as np
import scipy.optimize
from astropy.coordinates import Angle
from astropy.utils import lazyproperty
from gammapy.datasets.map import MapEvaluator
from gammapy.maps import Map, WcsGeom
from gammapy.modeling.models import (
    PointSpatialModel, PowerLawSpectralModel, SkyModel, ConstantFluxSpatialModel
)
from gammapy.stats import (
    amplitude_bounds_cython,
    cash,
    cash_sum_cython,
    f_cash_root_cython,
)
from gammapy.utils.array import shape_2N, symmetric_crop_pad_width
from .core import Estimator

__all__ = ["TSMapEstimator"]

log = logging.getLogger(__name__)

FLUX_FACTOR = 1e-12
MAX_NITER = 20
RTOL = 1e-3


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


def f_cash(x, counts, background, model):
    """Wrapper for cash statistics, that defines the model function.

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count image slice, where model is defined.
    background : `~numpy.ndarray`
        Background image slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    return cash_sum_cython(
        counts.ravel(), (background + x * FLUX_FACTOR * model).ravel()
    )


class TSMapEstimator(Estimator):
    r"""Compute TS map from a MapDataset using different optimization methods.

    The map is computed fitting by a single parameter amplitude fit. The fit is
    simplified by finding roots of the the derivative of the fit statistics using
    various root finding algorithms. The approach is described in Appendix A
    in Stewart (2009).

    Parameters
    ----------
    model : `~gammapy.modeling.model.SkyModel`
        Source model kernel. If set to None, assume point source model, PointSpatialModel.
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
        the amplitude fit.
    selection_optional : list of str or 'all'
        Which maps to compute besides delta TS, significance, flux and symmetric error on flux.
        Available options are:

            * "errn-errp": estimate assymmetric error on flux.
            * "ul": estimate upper limits on flux.

        By default all steps are executed.

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

    Where :math:`F` is the fitted flux amplitude.

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
    ):
        self.kernel_width = Angle(kernel_width)

        if model is None:
            model = SkyModel(
                spectral_model=PowerLawSpectralModel(),
                spatial_model=PointSpatialModel(),
            )

        self.model = model
        self.downsampling_factor = downsampling_factor
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.threshold = threshold
        self.rtol = rtol

        self.selection_optional = selection_optional
        self._flux_estimator = BrentqFluxEstimator(
            rtol=self.rtol,
            n_sigma=self.n_sigma,
            n_sigma_ul=self.n_sigma_ul,
            selection_optional=selection_optional,
            ts_threshold=threshold
        )

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

        axis = dataset.exposure.geom.get_axis_by_name("energy_true")

        geom = WcsGeom.create(
            skydir=model.position, proj="TAN", npix=npix, axes=[axis], binsz=binsz
        )

        exposure = Map.from_geom(geom, unit="cm2 s1")
        exposure.data += 1.0

        # We use global evaluation mode to not modify the geometry
        evaluator = MapEvaluator(model, evaluation_mode="global")
        evaluator.update(exposure, dataset.psf, dataset.edisp, dataset.counts.geom)

        kernel = evaluator.compute_npred()
        kernel.data /= kernel.data.sum()

        if (self.kernel_width > geom.width).any():
            raise ValueError(
                "Kernel shape larger than map shape, please adjust"
                " size of the kernel"
            )
        return kernel

    def estimate_exposure(self, dataset):
        """Estimate exposure map in reco energy


        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.

        Returns
        -------
        exposure : `Map`
            Exposure map

        """
        # TODO: clean this up a bit...
        models = dataset.models

        model = SkyModel(
            spectral_model=self.model.spectral_model,
            spatial_model=ConstantFluxSpatialModel(),
        )
        model.apply_irf["psf"] = False

        energy_axis = dataset.exposure.geom.get_axis_by_name("energy_true")
        energy = energy_axis.edges

        flux = model.spectral_model.integral(
            emin=energy.min(), emax=energy.max()
        )
        dataset.models = [model]
        npred = dataset.npred()
        dataset.models = models
        data = (npred.data / flux).to("cm2 s")
        return npred.copy(data=data.value, unit=data.unit)

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
            exposure = self.estimate_exposure(dataset)

        kernel = kernel / np.sum(kernel ** 2)
        flux = (dataset.counts - dataset.npred()) / exposure
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

        mask &= dataset.mask_safe.reduce_over_axes(
                func=np.logical_or, keepdims=False
            )

        # in some image there are pixels, which have exposure, but zero
        # background, which doesn't make sense and causes the TS computation
        # to fail, this is a temporary fix
        background = dataset.npred().sum_over_axes(keepdims=False)
        mask[background.data == 0] = False
        return Map.from_geom(data=mask, geom=geom)

    @staticmethod
    def estimate_sqrt_ts(map_ts):
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
        with np.errstate(invalid="ignore", divide="ignore"):
            ts = map_ts.data
            sqrt_ts = np.where(ts > 0, np.sqrt(ts), -np.sqrt(-ts))
        return map_ts.copy(data=sqrt_ts)

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
        if self.downsampling_factor:
            shape = dataset.counts.geom.to_image().data_shape
            pad_width = symmetric_crop_pad_width(shape, shape_2N(shape))[0]
            dataset = dataset.pad(pad_width).downsample(self.downsampling_factor)

        # First create 2D map arrays
        counts = dataset.counts
        background = dataset.npred()

        exposure = self.estimate_exposure(dataset)

        kernel = self.estimate_kernel(dataset)

        mask = self.estimate_mask_default(dataset, kernel.data)

        flux = self.estimate_flux_default(dataset, kernel.data, exposure=exposure)

        wrap = functools.partial(
            _ts_value,
            counts=counts.data.astype(float),
            exposure=exposure.data.astype(float),
            background=background.data.astype(float),
            kernel=kernel.data,
            flux=flux.data,
            flux_estimator=self._flux_estimator
        )

        x, y = np.where(np.squeeze(mask.data))
        positions = list(zip(x, y))
        results = list(map(wrap, positions))

        # Set TS values at given positions
        j, i = zip(*positions)

        names = ["ts", "flux", "niter", "flux_err"]

        if "errn-errp" in self.selection_optional:
            names += ["flux_errp", "flux_errn"]

        if "ul" in self.selection_optional:
            names += ["flux_ul"]

        result = {}

        geom = counts.geom.to_image()

        for name in names:
            unit = 1 / exposure.unit if "flux" in name else ""
            m = Map.from_geom(geom=geom, data=np.nan, unit=unit)
            m.data[j, i] = [_[name] for _ in results]
            result[name] = m

        result["sqrt_ts"] = self.estimate_sqrt_ts(result["ts"])

        if self.downsampling_factor:
            for name in names:
                order = 0 if name == "niter" else 1
                result[name] = result[name].upsample(
                    factor=self.downsampling_factor, preserve_counts=False, order=order
                )
                result[name] = result[name].crop(crop_width=pad_width)

        return result


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
    def __init__(self, model, counts, background, x_guess):
        self.model = model
        self.counts = counts
        self.background = background
        self.x_guess = x_guess
        self.FLUX_FACTOR = 1e-12

    @lazyproperty
    def x_bounds(self):
        """Bounds for x"""
        x_min, x_max, x_min_total = amplitude_bounds_cython(
            self.counts.ravel(), self.background.ravel(), self.model.ravel()
        )
        return x_min / self.FLUX_FACTOR, x_max / self.FLUX_FACTOR, x_min_total / self.FLUX_FACTOR

    def npred(self, x):
        """Predicted number of counts"""
        return self.background + x * self.FLUX_FACTOR * self.model

    def stat_sum(self, x):
        """Stat sum"""
        return cash_sum_cython(
            self.counts.ravel(), self.npred(x).ravel()
        )

    def stat_derivative(self, x):
        """Stat derivative"""
        return f_cash_root_cython(
            x * self.FLUX_FACTOR, self.counts.ravel(), self.background.ravel(), self.model.ravel()
        ) * self.FLUX_FACTOR

    def stat_2nd_derivative(self, x):
        """Stat 2nd derivative"""
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.model ** 2 * self.counts / (self.background + x * self.FLUX_FACTOR * self.model) ** 2).sum()

    @classmethod
    def from_arrays(cls, counts, background, exposure, flux, position, kernel):
        """"""
        counts_cutout = _extract_array(counts, kernel.shape, position)
        background_cutout = _extract_array(background, kernel.shape, position)
        exposure_cutout = _extract_array(exposure, kernel.shape, position)
        x_guess = flux[position]
        return cls(
            counts=counts_cutout,
            background=background_cutout,
            model=kernel * exposure_cutout,
            x_guess=x_guess
        )


# TODO: merge with `FluxEstimator`?
class BrentqFluxEstimator(Estimator):
    """Single parameter flux estimator"""
    _available_selection_optional = ["errn-errp", "ul"]
    tag = "BrentqFluxEstimator"

    def __init__(self, rtol, n_sigma, n_sigma_ul, selection_optional=None, max_niter=20, ts_threshold=None):
        self.rtol = rtol
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.selection_optional = selection_optional
        self.max_niter = max_niter
        self.ts_threshold = ts_threshold

    def estimate_best_fit(self, dataset):
        """Optimize for a single parameter"""
        result = {}
        # Compute amplitude bounds and assert counts > 0
        amplitude_min, amplitude_max, amplitude_min_total = dataset.x_bounds

        if not dataset.counts.sum() > 0:
            amplitude, niter = amplitude_min_total, 0

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result_fit = scipy.optimize.brentq(
                        dataset.stat_derivative,
                        amplitude_min,
                        amplitude_max,
                        maxiter=self.max_niter,
                        full_output=True,
                        rtol=self.rtol,
                    )
                    amplitude = max(result_fit[0], amplitude_min_total)
                    niter = result_fit[1].iterations
                except (RuntimeError, ValueError):
                    # Where the root finding fails NaN is set as amplitude
                    amplitude, niter = np.nan, self.max_niter

        stat = dataset.stat_sum(x=amplitude)
        stat_null = dataset.stat_sum(x=0)
        result["ts"] = (stat_null - stat) * np.sign(amplitude)
        result["flux"] = amplitude * dataset.FLUX_FACTOR
        result["niter"] = niter
        result["flux_err"] = np.sqrt(1 / dataset.stat_2nd_derivative(amplitude)) * self.n_sigma
        result["stat"] = stat
        return result

    @property
    def nan_result(self):
        result = {
            "flux": np.nan,
            "stat": np.nan,
            "success": False,
            "flux_err": np.nan,
            "ts": np.nan,
            "niter": 0
        }

        if "errn-errp" in self.selection_optional:
            result.update({"flux_errp": np.nan, "flux_errn": np.nan})

        if "ul" in self.selection_optional:
            result.update({"flux_ul": np.nan})

        return result

    def _confidence(self, dataset, n_sigma, result, positive):

        stat_best = result["stat"]
        amplitude = result["flux"] / FLUX_FACTOR
        amplitude_err = result["flux_err"] / FLUX_FACTOR

        def ts_diff(x):
            return (stat_best + n_sigma ** 2) - dataset.stat_sum(x)

        if positive:
            min_amplitude = amplitude
            max_amplitude = amplitude + 1e2 * amplitude_err
            factor = 1
        else:
            min_amplitude = amplitude - 1e2 * amplitude_err
            max_amplitude = amplitude
            factor = -1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result_fit = scipy.optimize.brentq(
                    ts_diff,
                    min_amplitude,
                    max_amplitude,
                    maxiter=self.max_niter,
                    rtol=self.rtol,
                )
                return (result_fit - amplitude) * factor * FLUX_FACTOR
            except (RuntimeError, ValueError):
                # Where the root finding fails NaN is set as amplitude
                return np.nan

    def estimate_ul(self, dataset, result):
        """"""

        flux_ul = self._confidence(
            dataset=dataset, n_sigma=self.n_sigma_ul, result=result, positive=True
        )

        return {"flux_ul": flux_ul}

    def estimate_errn_errp(self, dataset, result):
        """
        Compute amplitude errors using likelihood profile method.
        """

        flux_errn = self._confidence(
            dataset=dataset, result=result, n_sigma=self.n_sigma, positive=False
        )
        flux_errp = self._confidence(
            dataset=dataset, result=result, n_sigma=self.n_sigma, positive=True
        )
        return {"flux_errn": flux_errn, "flux_errp": flux_errp}

    def run(self, dataset):
        """"""
        if self.ts_threshold is not None:
            flux = dataset.x_guess
            stat = dataset.stat_sum(x=flux / FLUX_FACTOR)
            stat_null = dataset.stat_sum(x=0)
            ts = (stat_null - stat) * np.sign(flux)
            if ts < self.ts_threshold:
                return self.nan_result

        result = self.estimate_best_fit(dataset)

        if "ul" in self.selection_optional:
            result.update(self.estimate_ul(dataset, result))

        if "errn-errp" in self.selection_optional:
            result.update(self.estimate_errn_errp(dataset, result))

        return result


def _ts_value(
    position,
    counts,
    exposure,
    background,
    kernel,
    flux,
    flux_estimator
):
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
    flux : `~numpy.ndarray`
        Flux image. The flux value at the given pixel position is used as
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
        flux=flux
    )

    return flux_estimator.run(dataset)

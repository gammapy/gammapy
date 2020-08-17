# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to compute TS images."""
import functools
import logging
import warnings
import numpy as np
import scipy.optimize
from astropy.coordinates import Angle
from gammapy.datasets.map import MapEvaluator
from gammapy.maps import Map, WcsGeom
from gammapy.modeling.models import PointSpatialModel, PowerLawSpectralModel, SkyModel
from gammapy.stats import (
    amplitude_bounds_cython,
    cash,
    cash_sum_cython,
    f_cash_root_cython,
)
from gammapy.makers.utils import _map_spectrum_weight
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
    x_width = shape[1] // 2
    y_width = shape[0] // 2
    y_lo = position[0] - y_width
    y_hi = position[0] + y_width + 1
    x_lo = position[1] - x_width
    x_hi = position[1] + x_width + 1
    return array[y_lo:y_hi, x_lo:x_hi]


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
    selection : list of str or 'all'
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
    _selection_base = ["flux", "flux_err", "ts", "sqrt_ts", "niter"]
    _available_selection_optional = ["errn-errp", "ul"]

    def __init__(
        self,
        model=None,
        kernel_width="0.2 deg",
        downsampling_factor=None,
        n_sigma=1,
        n_sigma_ul=2,
        threshold=None,
        rtol=0.001,
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

    def get_kernel(self, dataset):
        """Set the convolution kernel for the input dataset.

        Convolves the model with the PSFKernel at the center of the dataset.
        If no PSFMap or PSFKernel is found the dataset, the model is used without convolution.
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

        kernel = evaluator.compute_npred().sum_over_axes()
        kernel.data /= kernel.data.sum()

        if (self.kernel_width > geom.width).any():
            raise ValueError(
                "Kernel shape larger than map shape, please adjust"
                " size of the kernel"
            )
        return kernel

    @staticmethod
    def flux_default(dataset, kernel):
        """Estimate default flux map using a given kernel.

        Parameters
        ----------
        dataset : `~gammapy.cube.MapDataset`
            Input dataset.
        kernel : `~numpy.ndarray`
            Source model kernel.

        Returns
        -------
        flux_approx : `~gammapy.maps.WcsNDMap`
            Approximate flux map (2D).
        """
        flux = dataset.counts - dataset.npred()
        flux = flux.sum_over_axes(keepdims=False)
        flux /= dataset.exposure.sum_over_axes(keepdims=False)
        flux /= np.sum(kernel ** 2)
        return flux.convolve(kernel)

    @staticmethod
    def mask_default(exposure, background, kernel):
        """Compute default mask where to estimate TS values.

        Parameters
        ----------
        exposure : `~gammapy.maps.Map`
            Input exposure map.
        background : `~gammapy.maps.Map`
            Input background map.
        kernel : `~numpy.ndarray`
            Source model kernel.

        Returns
        -------
        mask : `gammapy.maps.WcsNDMap`
            Mask map.
        """
        mask = np.zeros(exposure.data.shape, dtype=int)

        # mask boundary
        slice_x = slice(kernel.shape[1] // 2, -kernel.shape[1] // 2 + 1)
        slice_y = slice(kernel.shape[0] // 2, -kernel.shape[0] // 2 + 1)
        mask[slice_y, slice_x] = 1

        # positions where exposure == 0 are not processed
        mask &= exposure.data > 0

        # in some image there are pixels, which have exposure, but zero
        # background, which doesn't make sense and causes the TS computation
        # to fail, this is a temporary fix
        mask[background.data == 0] = 0

        return exposure.copy(data=mask.astype("int"), unit="")

    @staticmethod
    def sqrt_ts(map_ts):
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
        counts = dataset.counts.sum_over_axes(keepdims=False)
        background = dataset.npred().sum_over_axes(keepdims=False)

        exposure = _map_spectrum_weight(dataset.exposure, self.model.spectral_model)
        exposure = exposure.sum_over_axes(keepdims=False)

        kernel = self.get_kernel(dataset)

        if dataset.mask is not None:
            mask = counts.copy(data=(dataset.mask.sum(axis=0) > 0).astype("int"))
        else:
            mask = counts.copy(data=np.ones_like(counts).astype("int"))

        mask.data &= self.mask_default(exposure, background, kernel.data).data

        keys = ["ts", "sqrt_ts", "flux", "niter", "flux_err"]

        if "errn-errp" in self.selection_optional:
            keys.append("flux_errp")
            keys.append("flux_errn")

        if "ul" in self.selection_optional:
            keys.append("flux_ul")

        result = {}
        for name in keys:
            unit = 1 / exposure.unit if "flux" in name else ""
            data = np.nan * np.ones_like(counts.data)
            result[name] = counts.copy(data=data, unit=unit)

        flux_map = self.flux_default(dataset, kernel.data)

        if self.threshold:
            flux = flux_map.data
        else:
            flux = None

        # prepare dtype for cython methods
        counts_array = counts.data.astype(float)
        background_array = background.data.astype(float)
        exposure_array = exposure.data.astype(float)

        # Compute null statistics per pixel for the whole image
        c_0 = cash(counts_array, background_array)

        compute_errn_errp = True if "errn-errp" in self.selection_optional else False
        compute_ul = True if "ul" in self.selection_optional else False

        wrap = functools.partial(
            _ts_value,
            counts=counts_array,
            exposure=exposure_array,
            background=background_array,
            c_0=c_0,
            kernel=kernel.data,
            flux=flux,
            compute_errn_errp=compute_errn_errp,
            compute_ul=compute_ul,
            threshold=self.threshold,
            n_sigma=self.n_sigma,
            n_sigma_ul=self.n_sigma_ul,
            rtol=self.rtol,
        )

        x, y = np.where(np.squeeze(mask.data))
        positions = list(zip(x, y))
        results = list(map(wrap, positions))

        # Set TS values at given positions
        j, i = zip(*positions)

        names = ["ts", "flux", "niter", "flux_err"]\

        if "errn-errp" in self.selection_optional:
            names += ["flux_errp", "flux_errn"]

        if "ul" in self.selection_optional:
            names += ["flux_ul"]

        for name in names:
            result[name].data[j, i] = [_[name] for _ in results]

        result["sqrt_ts"] = self.sqrt_ts(result["ts"])

        if self.downsampling_factor:
            for name in keys:
                order = 0 if name == "niter" else 1
                result[name] = result[name].upsample(
                    factor=self.downsampling_factor, preserve_counts=False, order=order
                )
                result[name] = result[name].crop(crop_width=pad_width)

        return result


def _ts_value(
    position,
    counts,
    exposure,
    background,
    c_0,
    kernel,
    flux,
    compute_errn_errp,
    compute_ul,
    n_sigma,
    n_sigma_ul,
    threshold,
    rtol,
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
    # Get data slices
    counts_ = _extract_array(counts, kernel.shape, position)
    background_ = _extract_array(background, kernel.shape, position)
    exposure_ = _extract_array(exposure, kernel.shape, position)
    c_0_ = _extract_array(c_0, kernel.shape, position)

    model = exposure_ * kernel

    c_0 = c_0_.sum()

    if threshold is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            amplitude = flux[position]
            c_1 = f_cash(amplitude / FLUX_FACTOR, counts_, background_, model)
        # Don't fit if pixel significance is low
        if c_0 - c_1 < threshold:
            result = {}
            result["ts"] = (c_0 - c_1) * np.sign(amplitude)
            result["flux"] = amplitude
            result["niter"] = 0
            result["flux_err"] = np.nan
            if compute_errn_errp:
                result["flux_errn"] = np.nan
                result["flux_errp"] = np.nan
            if compute_ul:
                result["flux_ul"] = np.nan
            return result

    amplitude, niter = _root_amplitude_brentq(counts_, background_, model, rtol=rtol)

    with np.errstate(invalid="ignore", divide="ignore"):
        c_1 = f_cash(amplitude, counts_, background_, model)

    result = {}
    result["ts"] = (c_0 - c_1) * np.sign(amplitude)
    result["flux"] = amplitude * FLUX_FACTOR
    result["niter"] = niter

    flux_err = _compute_flux_err_covar(amplitude, counts_, background_, model)
    result["flux_err"] = flux_err * n_sigma

    if compute_errn_errp:
        flux_errp = _compute_flux_err_conf(
            amplitude, counts_, background_, model, c_1, n_sigma, True
        )
        result["flux_errp"] = FLUX_FACTOR * flux_errp

        flux_errn = _compute_flux_err_conf(
            amplitude, counts_, background_, model, c_1, n_sigma, False
        )
        result["flux_errn"] = FLUX_FACTOR * flux_errn

    if compute_ul:
        flux_ul = _compute_flux_err_conf(
            amplitude, counts_, background_, model, c_1, n_sigma_ul, True
        )
        result["flux_ul"] = FLUX_FACTOR * flux_ul + result["flux"]
    return result


def _root_amplitude_brentq(counts, background, model, rtol=RTOL):
    """Fit amplitude by finding roots using Brent algorithm.

    See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Slice of count image
    background : `~numpy.ndarray`
        Slice of background image
    model : `~numpy.ndarray`
        Model template to fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    # Compute amplitude bounds and assert counts > 0
    bounds = amplitude_bounds_cython(counts.ravel(), background.ravel(), model.ravel())
    amplitude_min, amplitude_max, amplitude_min_total = bounds

    if not counts.sum() > 0:
        return amplitude_min_total, 0

    args = (counts.ravel(), background.ravel(), model.ravel())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = scipy.optimize.brentq(
                f_cash_root_cython,
                amplitude_min,
                amplitude_max,
                args=args,
                maxiter=MAX_NITER,
                full_output=True,
                rtol=rtol,
            )
            return max(result[0], amplitude_min_total), result[1].iterations
        except (RuntimeError, ValueError):
            # Where the root finding fails NaN is set as amplitude
            return np.nan, MAX_NITER


def _compute_flux_err_covar(x, counts, background, model):
    """
    Compute amplitude errors using inverse 2nd derivative method.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = (model ** 2 * counts) / (background + x * FLUX_FACTOR * model) ** 2
        return np.sqrt(1.0 / stat.sum())


def _compute_flux_err_conf(
    amplitude, counts, background, model, c_1, n_sigma, positive=True
):
    """
    Compute amplitude errors using likelihood profile method.
    """

    def ts_diff(x, counts, background, model):
        return (c_1 + n_sigma ** 2) - f_cash(x, counts, background, model)

    bounds = amplitude_bounds_cython(counts.ravel(), background.ravel(), model.ravel())
    amplitude_min, amplitude_max, _ = bounds

    args = (counts, background, model)

    if positive:
        min_amplitude = amplitude
        max_amplitude = amplitude_max
        factor = 1
    else:
        min_amplitude = amplitude_min
        max_amplitude = amplitude
        factor = -1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = scipy.optimize.brentq(
                ts_diff,
                min_amplitude,
                max_amplitude,
                args=args,
                maxiter=MAX_NITER,
                rtol=1e-3,
            )
            return (result - amplitude) * factor
        except (RuntimeError, ValueError):
            # Where the root finding fails NaN is set as amplitude
            return np.nan

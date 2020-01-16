# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to compute TS images."""
import functools
import logging
import warnings
import numpy as np
import scipy.optimize
from astropy.convolution import CustomKernel, Kernel2D
from gammapy.stats import cash, cash_sum_cython
from gammapy.utils.array import shape_2N, symmetric_crop_pad_width
from ._test_statistics_cython import (
    _amplitude_bounds_cython,
    _f_cash_root_cython,
    _x_best_leastsq,
)

__all__ = ["TSMapEstimator"]

log = logging.getLogger(__name__)

FLUX_FACTOR = 1e-12
MAX_NITER = 20
RTOL = 1e-3


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


class TSMapEstimator:
    r"""Compute TS map from a MapDataset using different optimization methods.

    The map is computed fitting by a single parameter amplitude fit. The fit is
    simplified by finding roots of the the derivative of the fit statistics using
    various root finding algorithms. The approach is sescribed in Appendix A
    in Stewart (2009).

    Parameters
    ----------
    method : str ('root')
        The following options are available:

        * ``'root brentq'`` (default)
            Fit amplitude by finding the roots of the the derivative of the fit
            statistics using the brentq method.
        * ``'root newton'``
            Fit amplitude by finding the roots of the the derivative of the fit
            statistics using Newton's method.
        * ``'leastsq iter'``
            Fit the amplitude by an iterative least square fit, that can be solved
            analytically.
    error_method : ['covar', 'conf']
        Error estimation method.
    error_sigma : int (1)
        Sigma for flux error.
    ul_method : ['covar', 'conf']
        Upper limit estimation method.
    ul_sigma : int (2)
        Sigma for flux upper limits.
    threshold : float (None)
        If the TS value corresponding to the initial flux estimate is not above
        this threshold, the optimizing step is omitted to save computing time.
    rtol : float (0.001)
        Relative precision of the flux estimate. Used as a stopping criterion for
        the amplitude fit.

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

    def __init__(
        self,
        method="root brentq",
        error_method="covar",
        error_sigma=1,
        ul_method="covar",
        ul_sigma=2,
        threshold=None,
        rtol=0.001,
    ):

        if method not in ["root brentq", "root newton", "leastsq iter"]:
            raise ValueError(f"Not a valid method: '{method}'")

        if error_method not in ["covar", "conf"]:
            raise ValueError(f"Not a valid error method '{error_method}'")

        self.parameters = {
            "method": method,
            "error_method": error_method,
            "error_sigma": error_sigma,
            "ul_method": ul_method,
            "ul_sigma": ul_sigma,
            "threshold": threshold,
            "rtol": rtol,
        }

    @staticmethod
    def flux_default(dataset, kernel):
        """Estimate default flux map using a given kernel.

        Parameters
        ----------
        dataset : `~gammapy.cube.MapDataset`
            Input dataset.
        kernel : `~stropy.convolution.Kernel2D`
            Source model kernel.

        Returns
        -------
        flux_approx : `~gammapy.maps.WcsNDMap`
            Approximate flux map (2D).
        """
        flux = dataset.counts - dataset.npred()
        flux = flux.sum_over_axes(keepdims=False)
        flux /= dataset.exposure.sum_over_axes(keepdims=False)
        flux /= np.sum(kernel.array ** 2)
        return flux.convolve(kernel.array)

    @staticmethod
    def mask_default(exposure, background, kernel):
        """Compute default mask where to estimate TS values.

        Parameters
        ----------
        exposure : `~gammapy.maps.Map`
            Input exposure map.
        background : `~gammapy.maps.Map`
            Input background map.
        kernel : `astropy.convolution.Kernel2D`
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
        mask[background == 0] = 0

        return exposure.copy(data=mask.astype("int"))

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

    def run(self, dataset, kernel, which="all", downsampling_factor=None):
        """
        Run TS map estimation.

        Requires a MapDataset with counts, exposure and background_model
        properly set to run.

        Parameters
        ----------
        kernel : `astropy.convolution.Kernel2D` or 2D `~numpy.ndarray`
            Source model kernel.
        which : list of str or 'all'
            Which maps to compute.
        downsampling_factor : int
            Sample down the input maps to speed up the computation. Only integer
            values that are a multiple of 2 are allowed. Note that the kernel is
            not sampled down, but must be provided with the downsampled bin size.

        Returns
        -------
        maps : dict
            Result maps.
        """
        p = self.parameters

        if (np.array(kernel.shape) > np.array(dataset.counts.data.shape[1:])).any():
            raise ValueError(
                "Kernel shape larger than map shape, please adjust"
                " size of the kernel"
            )

        # First create 2D map arrays
        counts = dataset.counts.sum_over_axes(keepdims=False)
        background = dataset.npred().sum_over_axes(keepdims=False)
        exposure = dataset.exposure.sum_over_axes(keepdims=False)
        if dataset.mask is not None:
            mask = counts.copy(data=(dataset.mask.sum(axis=0) > 0).astype("int"))
        else:
            mask = counts.copy(data=np.ones_like(counts).astype("int"))

        if downsampling_factor:
            shape = counts.data.shape
            pad_width = symmetric_crop_pad_width(shape, shape_2N(shape))[0]

            counts = counts.pad(pad_width).downsample(
                downsampling_factor, preserve_counts=True
            )
            background = background.pad(pad_width).downsample(
                downsampling_factor, preserve_counts=True
            )
            exposure = exposure.pad(pad_width).downsample(
                downsampling_factor, preserve_counts=False
            )
            mask = mask.pad(pad_width).downsample(
                downsampling_factor, preserve_counts=False
            )
            mask.data = mask.data.astype("int")

        mask.data &= self.mask_default(exposure, background, kernel).data

        if not isinstance(kernel, Kernel2D):
            kernel = CustomKernel(kernel)

        if which == "all":
            which = ["ts", "sqrt_ts", "flux", "flux_err", "flux_ul", "niter"]

        result = {}
        for name in which:
            data = np.nan * np.ones_like(counts.data)
            result[name] = counts.copy(data=data)

        if p["threshold"] or p["method"] == "root newton":
            flux = self.flux_default(dataset, kernel).data
        else:
            flux = None

        # prepare dtype for cython methods
        counts_array = counts.data.astype(float)
        background_array = background.data.astype(float)
        exposure_array = exposure.data.astype(float)

        # Compute null statistics per pixel for the whole image
        c_0 = cash(counts_array, background_array)

        error_method = p["error_method"] if "flux_err" in which else "none"
        ul_method = p["ul_method"] if "flux_ul" in which else "none"

        wrap = functools.partial(
            _ts_value,
            counts=counts_array,
            exposure=exposure_array,
            background=background_array,
            c_0=c_0,
            kernel=kernel,
            flux=flux,
            method=p["method"],
            error_method=error_method,
            threshold=p["threshold"],
            error_sigma=p["error_sigma"],
            ul_method=ul_method,
            ul_sigma=p["ul_sigma"],
            rtol=p["rtol"],
        )

        x, y = np.where(np.squeeze(mask.data))
        positions = list(zip(x, y))
        results = list(map(wrap, positions))

        # Set TS values at given positions
        j, i = zip(*positions)
        for name in ["ts", "flux", "niter"]:
            result[name].data[j, i] = [_[name] for _ in results]

        if "flux_err" in which:
            result["flux_err"].data[j, i] = [_["flux_err"] for _ in results]

        if "flux_ul" in which:
            result["flux_ul"].data[j, i] = [_["flux_ul"] for _ in results]

        # Compute sqrt(TS) values
        if "sqrt_ts" in which:
            result["sqrt_ts"] = self.sqrt_ts(result["ts"])

        if downsampling_factor:
            for name in which:
                order = 0 if name == "niter" else 1
                result[name] = result[name].upsample(
                    factor=downsampling_factor, preserve_counts=False, order=order
                )
                result[name] = result[name].crop(crop_width=pad_width)

        return result

    def __repr__(self):
        p = self.parameters
        info = self.__class__.__name__
        info += "\n\nParameters:\n\n"
        for key in p:
            info += f"\t{key:13s}: {p[key]}\n"
        return info


def _ts_value(
    position,
    counts,
    exposure,
    background,
    c_0,
    kernel,
    flux,
    method,
    error_method,
    error_sigma,
    ul_method,
    ul_sigma,
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

    model = exposure_ * kernel._array

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
            result["flux_ul"] = np.nan
            return result

    if method == "root brentq":
        amplitude, niter = _root_amplitude_brentq(
            counts_, background_, model, rtol=rtol
        )
    elif method == "root newton":
        amplitude, niter = _root_amplitude(
            counts_, background_, model, flux[position], rtol=rtol
        )
    elif method == "leastsq iter":
        amplitude, niter = _leastsq_iter_amplitude(
            counts_, background_, model, rtol=rtol
        )
    else:
        raise ValueError(f"Invalid method: {method}")

    with np.errstate(invalid="ignore", divide="ignore"):
        c_1 = f_cash(amplitude, counts_, background_, model)

    result = {}
    result["ts"] = (c_0 - c_1) * np.sign(amplitude)
    result["flux"] = amplitude * FLUX_FACTOR
    result["niter"] = niter

    if error_method == "covar":
        flux_err = _compute_flux_err_covar(amplitude, counts_, background_, model)
        result["flux_err"] = flux_err * error_sigma
    elif error_method == "conf":
        flux_err = _compute_flux_err_conf(
            amplitude, counts_, background_, model, c_1, error_sigma
        )
        result["flux_err"] = FLUX_FACTOR * flux_err

    if ul_method == "covar":
        result["flux_ul"] = result["flux"] + ul_sigma * result["flux_err"]
    elif ul_method == "conf":
        flux_ul = _compute_flux_err_conf(
            amplitude, counts_, background_, model, c_1, ul_sigma
        )
        result["flux_ul"] = FLUX_FACTOR * flux_ul + result["flux"]
    return result


def _leastsq_iter_amplitude(counts, background, model, maxiter=MAX_NITER, rtol=RTOL):
    """Fit amplitude using an iterative least squares algorithm.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Slice of counts image
    background : `~numpy.ndarray`
        Slice of background image
    model : `~numpy.ndarray`
        Model template to fit.
    maxiter : int
        Maximum number of iterations.
    rtol : float
        Relative flux error.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    bounds = _amplitude_bounds_cython(counts, background, model)
    amplitude_min, amplitude_max, amplitude_min_total = bounds

    if not counts.sum() > 0:
        return amplitude_min_total, 0

    weights = np.ones(model.shape)

    x_old = 0
    for i in range(maxiter):
        x = _x_best_leastsq(counts, background, model, weights)
        if abs((x - x_old) / x) < rtol:
            return max(x / FLUX_FACTOR, amplitude_min_total), i + 1
        else:
            weights = x * model + background
            x_old = x
    return max(x / FLUX_FACTOR, amplitude_min_total), MAX_NITER


def _root_amplitude(counts, background, model, flux, rtol=RTOL):
    """Fit amplitude by finding roots using newton algorithm.

    See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Slice of count image
    background : `~numpy.ndarray`
        Slice of background image
    model : `~numpy.ndarray`
        Model template to fit.
    flux : float
        Starting value for the fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    args = (counts, background, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return (
                scipy.optimize.newton(
                    _f_cash_root_cython, flux, args=args, maxiter=MAX_NITER, tol=rtol
                ),
                0,
            )
        except RuntimeError:
            # Where the root finding fails NaN is set as amplitude
            return np.nan, MAX_NITER


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
    bounds = _amplitude_bounds_cython(counts, background, model)
    amplitude_min, amplitude_max, amplitude_min_total = bounds

    if not counts.sum() > 0:
        return amplitude_min_total, 0

    args = (counts, background, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = scipy.optimize.brentq(
                _f_cash_root_cython,
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


def _compute_flux_err_conf(amplitude, counts, background, model, c_1, error_sigma):
    """
    Compute amplitude errors using likelihood profile method.
    """

    def ts_diff(x, counts, background, model):
        return (c_1 + error_sigma ** 2) - f_cash(x, counts, background, model)

    args = (counts, background, model)

    amplitude_max = amplitude + 1e4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = scipy.optimize.brentq(
                ts_diff,
                amplitude,
                amplitude_max,
                args=args,
                maxiter=MAX_NITER,
                rtol=1e-3,
            )
            return result - amplitude
        except (RuntimeError, ValueError):
            # Where the root finding fails NaN is set as amplitude
            return np.nan

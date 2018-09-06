# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Ring background estimation."""
from __future__ import absolute_import, division, print_function, unicode_literals
from itertools import product
import numpy as np
from astropy.convolution import Ring2DKernel, Tophat2DKernel
from astropy.coordinates import Angle
from ..image.utils import scale_cube

__all__ = ["AdaptiveRingBackgroundEstimator", "RingBackgroundEstimator"]


class AdaptiveRingBackgroundEstimator(object):
    """Adaptive ring background algorithm.

    This algorithm extends the `RingBackgroundEstimator` method by adapting the
    size of the ring to achieve a minimum on / off exposure ratio (alpha) in regions
    where the area to estimate the background from is limited.

    Parameters
    ----------
    r_in : `~astropy.units.Quantity`
        Inner radius of the ring.
    r_out_max : `~astropy.units.Quantity`
        Maximal outer radius of the ring.
    width : `~astropy.units.Quantity`
        Width of the ring.
    stepsize : `~astropy.units.Quantity`
        Stepsize used for increasing the radius.
    threshold_alpha : float
        Threshold on alpha above which the adaptive ring takes action.
    theta : `~astropy.units.Quantity`
        Integration radius used for alpha computation.
    method : {'fixed_width', 'fixed_r_in'}
        Adaptive ring method.

    Examples
    --------
    Example using `AdaptiveRingBackgroundEstimator`::

        from gammapy.maps import Map
        from gammapy.background import AdaptiveRingBackgroundEstimator

        filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz'
        images = {
            'counts': Map.read(filename, hdu='counts'),
            'exposure_on': Map.read(filename, hdu='exposure'),
            'exclusion': Map.read(filename, hdu='exclusion'),
        }

        adaptive_ring_bkg = AdaptiveRingBackgroundEstimator(
            r_in='0.22 deg',
            r_out_max='0.8 deg',
            width='0.1 deg',
        )
        results = adaptive_ring_bkg.run(images)
        results['background'].plot()

    See Also
    --------
    RingBackgroundEstimator, gammapy.detect.KernelBackgroundEstimator
    """

    def __init__(
        self,
        r_in,
        r_out_max,
        width,
        stepsize="0.02 deg",
        threshold_alpha=0.1,
        theta="0.22 deg",
        method="fixed_width",
    ):
        stepsize = Angle(stepsize)
        theta = Angle(theta)

        if method not in ["fixed_width", "fixed_r_in"]:
            raise ValueError("Not a valid adaptive ring method.")

        self._parameters = {
            "r_in": Angle(r_in),
            "r_out_max": Angle(r_out_max),
            "width": Angle(width),
            "stepsize": Angle(stepsize),
            "threshold_alpha": threshold_alpha,
            "theta": Angle(theta),
            "method": method,
        }

    @property
    def parameters(self):
        """Parameter dict."""
        return self._parameters

    def kernels(self, image):
        """Ring kernels according to the specified method.

        Parameters
        ----------
        image : `~gammapy.maps.WcsNDMap`
            Map specifying the WCS information.

        Returns
        -------
        kernels : list
            List of `~astropy.convolution.Ring2DKernel`
        """
        p = self.parameters

        scale = image.geom.pixel_scales[0].to("deg")
        r_in = p["r_in"].to("deg") / scale
        r_out_max = p["r_out_max"].to("deg") / scale
        width = p["width"].to("deg") / scale
        stepsize = p["stepsize"].to("deg") / scale

        if p["method"] == "fixed_width":
            r_ins = np.arange(r_in.value, (r_out_max - width).value, stepsize.value)
            widths = [width.value]
        elif p["method"] == "fixed_r_in":
            widths = np.arange(width.value, (r_out_max - r_in).value, stepsize.value)
            r_ins = [r_in.value]
        else:
            raise ValueError("Invalid method: {}".format(p["method"]))

        kernels = []
        for r_in, width in product(r_ins, widths):
            kernel = Ring2DKernel(r_in, width)
            kernel.normalize("peak")
            kernels.append(kernel)

        return kernels

    @staticmethod
    def _alpha_approx_cube(cubes):
        """Compute alpha as on_exposure / off_exposure.

        Where off_exposure < 0, alpha is set to infinity.
        """
        exposure_on = cubes["exposure_on"]
        exposure_off = cubes["exposure_off"]
        alpha_approx = np.where(exposure_off > 0, exposure_on / exposure_off, np.inf)
        return alpha_approx

    @staticmethod
    def _exposure_off_cube(exposure_on, exclusion, kernels):
        """Compute off exposure cube.

        The on exposure is convolved with different
        ring kernels and stacking the data along the third dimension.
        """
        exposure = exposure_on.data
        exclusion = exclusion.data
        return scale_cube(exposure * exclusion, kernels)

    def _exposure_on_cube(self, exposure_on, kernels):
        """Compute on exposure cube.

        Calculated by convolving the on exposure with a tophat
        of radius theta, and stacking all images along the third dimension.
        """
        scale = exposure_on.geom.pixel_scales[0].to("deg")
        theta = self.parameters["theta"] * scale

        tophat = Tophat2DKernel(theta.value)
        tophat.normalize("peak")
        exposure_on = exposure_on.convolve(tophat.array)
        exposure_on_cube = np.repeat(
            exposure_on.data[:, :, np.newaxis], len(kernels), axis=2
        )
        return exposure_on_cube

    @staticmethod
    def _off_cube(counts, exclusion, kernels):
        """Compute off cube.

        Calculated by convolving the raw counts with different ring kernels
        and stacking the data along the third dimension.
        """
        return scale_cube(counts.data * exclusion.data, kernels)

    def _reduce_cubes(self, cubes):
        """Compute off and off exposure map.

        Calulated by reducing the cubes. The data is
        iterated along the third axis (i.e. increasing ring sizes), the value
        with the first approximate alpha < threshold is taken.
        """
        threshold = self._parameters["threshold_alpha"]

        alpha_approx_cube = cubes["alpha_approx"]
        off_cube = cubes["off"]
        exposure_off_cube = cubes["exposure_off"]

        shape = alpha_approx_cube.shape[:2]
        off = np.tile(np.nan, shape)
        exposure_off = np.tile(np.nan, shape)

        for idx in np.arange(alpha_approx_cube.shape[-1]):
            mask = (alpha_approx_cube[:, :, idx] <= threshold) & np.isnan(off)
            off[mask] = off_cube[:, :, idx][mask]
            exposure_off[mask] = exposure_off_cube[:, :, idx][mask]

        return exposure_off, off

    def run(self, images):
        """Run adaptive ring background algorithm.

        Parameters
        ----------
        images : dict of `~gammapy.maps.WcsNDMap`
            Input sky maps.

        Returns
        -------
        result : dict of `~gammapy.maps.WcsNDMap`
            Result sky maps.
        """
        required = ["counts", "exposure_on", "exclusion"]
        counts, exposure_on, exclusion = [images[_] for _ in required]

        if not counts.geom.is_image:
            raise ValueError("Only 2D maps are supported")

        kernels = self.kernels(counts)
        cubes = {
            "exposure_on": self._exposure_on_cube(exposure_on, kernels),
            "exposure_off": self._exposure_off_cube(exposure_on, exclusion, kernels),
            "off": self._off_cube(counts, exclusion, kernels),
        }
        cubes["alpha_approx"] = self._alpha_approx_cube(cubes)

        exposure_off, off = self._reduce_cubes(cubes)
        alpha = exposure_on.data / exposure_off

        # set data outside fov to zero
        not_has_exposure = ~(exposure_on.data > 0)
        for data in [alpha, off, exposure_off]:
            data[not_has_exposure] = 0

        background = alpha * off

        return {
            "exposure_off": counts.copy(data=exposure_off),
            "off": counts.copy(data=off),
            "alpha": counts.copy(data=alpha),
            "background": counts.copy(data=background),
        }


class RingBackgroundEstimator(object):
    """Ring background method for cartesian coordinates.

    - Step 1: apply exclusion mask
    - Step 2: ring-correlate
    - Step 3: apply psi cut

    TODO: add method to apply the psi cut

    Parameters
    ----------
    r_in : `~astropy.units.Quantity`
        Inner ring radius
    width : `~astropy.units.Quantity`
        Ring width.
    use_fft_convolution : bool
        Use FFT convolution?


    Examples
    --------
    Example using `RingBackgroundEstimator`::

        from gammapy.maps import Map
        from gammapy.background import RingBackgroundEstimator

        filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz'
        images = {
            'counts': Map.read(filename, hdu='counts'),
            'exposure_on': Map.read(filename, hdu='exposure'),
            'exclusion': Map.read(filename, hdu='exclusion'),
        }

        ring_bkg = RingBackgroundEstimator(r_in='0.35 deg', width='0.3 deg')
        results = ring_bkg.run(images)
        results['background'].plot()

    See Also
    --------
    gammapy.detect.KernelBackgroundEstimator, AdaptiveRingBackgroundEstimator
    """

    def __init__(self, r_in, width, use_fft_convolution=False):
        self._parameters = {
            "r_in": Angle(r_in),
            "width": Angle(width),
            "use_fft_convolution": use_fft_convolution,
        }

    @property
    def parameters(self):
        """dict of parameters"""
        return self._parameters

    def kernel(self, image):
        """Ring kernel.

        Parameters
        ----------
        image : `~gammapy.maps.WcsNDMap`
            Input Map

        Returns
        -------
        ring : `~astropy.convolution.Ring2DKernel`
            Ring kernel.
        """
        p = self.parameters

        scale = image.geom.pixel_scales[0].to("deg")
        r_in = p["r_in"].to("deg") / scale
        width = p["width"].to("deg") / scale

        ring = Ring2DKernel(r_in.value, width.value)
        ring.normalize("peak")
        return ring

    def run(self, images):
        """Run ring background algorithm.

        Required Maps: {required}

        Parameters
        ----------
        images : dict of `~gammapy.maps.WcsNDMap`
            Input sky maps.

        Returns
        -------
        result : dict of `~gammapy.maps.WcsNDMap`
            Result sky maps
        """
        p = self.parameters
        required = ["counts", "exposure_on", "exclusion"]

        counts, exposure_on, exclusion = [images[_] for _ in required]

        result = {}
        ring = self.kernel(counts)

        counts_excluded = counts.copy(data=counts.data * exclusion.data.astype("float"))
        result["off"] = counts_excluded.convolve(
            ring.array, use_fft=p["use_fft_convolution"]
        )

        exposure_on_excluded = exposure_on.copy(
            data=exposure_on.data * exclusion.data.astype("float")
        )
        result["exposure_off"] = exposure_on_excluded.convolve(
            ring.array, use_fft=p["use_fft_convolution"]
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            # set pixels, where ring is too small to NaN
            not_has_off_exposure = ~(result["exposure_off"].data > 0)
            result["exposure_off"].data[not_has_off_exposure] = np.nan

            not_has_exposure = ~(exposure_on.data > 0)
            result["off"].data[not_has_exposure] = 0
            result["exposure_off"].data[not_has_exposure] = 0

            result["alpha"] = exposure_on.copy(
                data=exposure_on.data / result["exposure_off"].data
            )
            result["alpha"].data[not_has_exposure] = 0

        result["background"] = counts.copy(
            data=result["alpha"].data * result["off"].data
        )

        return result

    def __str__(self):
        s = "RingBackground parameters: \n"
        s += "r_in : {}\n".format(self.parameters["r_in"])
        s += "width: {}\n".format(self.parameters["width"])
        return s


def ring_r_out(theta, r_in, area_factor):
    """Compute ring outer radius.

    The determining equation is:
        area_factor =
        off_area / on_area =
        (pi (r_out**2 - r_in**2)) / (pi * theta**2 )

    Parameters
    ----------
    theta : float
        On region radius
    r_in : float
        Inner ring radius
    area_factor : float
        Desired off / on area ratio

    Returns
    -------
    r_out : float
        Outer ring radius
    """
    return np.sqrt(area_factor * theta ** 2 + r_in ** 2)


def ring_area_factor(theta, r_in, r_out):
    """Compute ring area factor.

    Parameters
    ----------
    theta : float
        On region radius
    r_in : float
        Inner ring radius
    r_out : float
        Outer ring radius
    """
    return (r_out ** 2 - r_in ** 2) / theta ** 2


def ring_alpha(theta, r_in, r_out):
    """Compute ring alpha, the inverse area factor.

    Parameters
    ----------
    theta : float
        On region radius
    r_in : float
        Inner ring radius
    r_out : float
        Outer ring radius
    """
    return 1. / ring_area_factor(theta, r_in, r_out)

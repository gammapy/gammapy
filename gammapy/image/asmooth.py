# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implementation of adaptive smoothing algorithms.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel
from ..stats import significance
from .utils import scale_cube

__all__ = ["ASmooth"]


def _significance_asmooth(counts, background):
    """Significance according to formula (5) in asmooth paper."""
    return (counts - background) / np.sqrt(counts + background)


class ASmooth(object):
    """Adaptively smooth counts image.

    Achieves a roughly constant significance of features across the whole image.

    Algorithm based on http://adsabs.harvard.edu/abs/2006MNRAS.368...65E

    The algorithm was slightly adapted to also allow Li & Ma and TS to estimate the
    significance of a feature in the image.

    Parameters
    ----------
    kernel : `astropy.convolution.Kernel`
        Smoothing kernel.
    method : {'simple', 'asmooth', 'lima'}
        Significance estimation method.
    threshold : float
        Significance threshold.
    scales : `~astropy.units.Quantity`
        Smoothing scales.
    """

    def __init__(
        self, kernel=Gaussian2DKernel, method="simple", threshold=5, scales=None
    ):
        self.parameters = {
            "kernel": kernel,
            "method": method,
            "threshold": threshold,
            "scales": scales,
        }

    def kernels(self, pixel_scale):
        """
        Ring kernels according to the specified method.

        Parameters
        ----------
        pixel_scale : `~astropy.coordinates.Angle`
            Sky image pixel scale

        Returns
        -------
        kernels : list
            List of `~astropy.convolution.Kernel`
        """
        p = self.parameters
        scales = p["scales"].to("deg") / Angle(pixel_scale).deg

        kernels = []
        for scale in scales.value:
            kernel = p["kernel"](scale, mode="oversample")
            # TODO: check if normalizing here makes sense
            kernel.normalize("peak")
            kernels.append(kernel)

        return kernels

    @staticmethod
    def _significance_cube(cubes, method):
        if method in {"lima", "simple"}:
            scube = significance(cubes["counts"], cubes["background"], method="lima")
        elif method == "asmooth":
            scube = _significance_asmooth(cubes["counts"], cubes["background"])
        elif method == "ts":
            raise NotImplementedError()
        else:
            raise ValueError(
                "Not a valid significance estimation method."
                " Choose one of the following: 'lima', 'simple',"
                " 'asmooth' or 'ts'"
            )
        return scube

    def run(self, counts, background=None, exposure=None):
        """
        Run image smoothing.

        Parameters
        ----------
        counts : `~gammapy.maps.WcsNDMap`
            Counts map
        background : `~gammapy.maps.WcsNDMap`
            Background map
        exposure : `~gammapy.maps.WcsNDMap`
            Exposure map

        Returns
        -------
        images : dict of `~gammapy.maps.WcsNDMap`
            Smoothed images; keys are:
                * 'counts'
                * 'background'
                * 'flux' (optional)
                * 'scales'
                * 'significance'.
        """
        from ..maps import WcsNDMap

        pixel_scale = counts.geom.pixel_scales.mean()
        kernels = self.kernels(pixel_scale)

        cubes = {}
        cubes["counts"] = scale_cube(counts.data, kernels)

        if background is not None:
            cubes["background"] = scale_cube(background.data, kernels)
        else:
            # TODO: Estimate background with asmooth method
            raise ValueError("Background estimation required.")

        if exposure is not None:
            flux = (counts.data - background.data) / exposure.data
            cubes["flux"] = scale_cube(flux, kernels)

        cubes["significance"] = self._significance_cube(
            cubes, method=self.parameters["method"]
        )

        smoothed = self._reduce_cubes(cubes, kernels)

        result = {}

        for key in ["counts", "background", "scale", "significance"]:
            data = smoothed[key]

            # set remaining pixels with significance < threshold to mean value
            if key in ["counts", "background"]:
                mask = np.isnan(data)
                data[mask] = np.mean(locals()[key].data[mask])
            result[key] = WcsNDMap(counts.geom, data)

        if exposure is not None:
            data = smoothed["flux"]
            mask = np.isnan(data)
            data[mask] = np.mean(flux[mask])
            result["flux"] = WcsNDMap(counts.geom, data)

        return result

    def _reduce_cubes(self, cubes, kernels):
        """
        Combine scale cube to image.

        Parameters
        ----------
        cubes : dict
            Data cubes
        """
        p = self.parameters
        shape = cubes["counts"].shape[:2]
        smoothed = {}

        # Init smoothed data arrays
        for key in ["counts", "background", "scale", "significance", "flux"]:
            smoothed[key] = np.tile(np.nan, shape)

        for idx, scale in enumerate(p["scales"]):
            # slice out 2D image at index idx out of cube
            slice_ = np.s_[:, :, idx]

            mask = np.isnan(smoothed["counts"])
            mask = (cubes["significance"][slice_] > p["threshold"]) & mask

            smoothed["scale"][mask] = scale
            smoothed["significance"][mask] = cubes["significance"][slice_][mask]

            # renormalize smoothed data arrays
            norm = kernels[idx].array.sum()
            for key in ["counts", "background"]:
                smoothed[key][mask] = cubes[key][slice_][mask] / norm
            if "flux" in cubes:
                smoothed["flux"][mask] = cubes["flux"][slice_][mask] / norm

        return smoothed

    @staticmethod
    def make_scales(n_scales, factor=np.sqrt(2), kernel=Gaussian2DKernel):
        """Create list of Gaussian widths."""
        if kernel == Gaussian2DKernel:
            sigma_0 = 1. / np.sqrt(9 * np.pi)
        elif kernel == Tophat2DKernel:
            sigma_0 = 1. / np.sqrt(np.pi)

        return sigma_0 * factor ** np.arange(n_scales)

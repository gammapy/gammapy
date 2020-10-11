# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Implementation of adaptive smoothing algorithms."""
import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel
from astropy.coordinates import Angle
from gammapy.datasets import MapDatasetOnOff, Datasets
from gammapy.maps import WcsNDMap, Map
from gammapy.stats import CashCountsStatistic
from gammapy.utils.pbar import pbar
from gammapy.utils.array import scale_cube
from gammapy.modeling.models import PowerLawSpectralModel
from .core import Estimator
from .utils import estimate_exposure_reco_energy

__all__ = ["ASmoothMapEstimator"]


def _sqrt_ts_asmooth(counts, background):
    """Significance according to formula (5) in asmooth paper."""
    return (counts - background) / np.sqrt(counts + background)


class ASmoothMapEstimator(Estimator):
    """Adaptively smooth counts image.

    Achieves a roughly constant sqrt_ts of features across the whole image.

    Algorithm based on https://ui.adsabs.harvard.edu/abs/2006MNRAS.368...65E

    The algorithm was slightly adapted to also allow Li & Ma  to estimate the
    sqrt_ts of a feature in the image.

    Parameters
    ----------
    scales : `~astropy.units.Quantity`
        Smoothing scales.
    kernel : `astropy.convolution.Kernel`
        Smoothing kernel.
    spectrum : `SpectralModel`
        Spectral model assumption
    method : {'asmooth', 'lima'}
        Significance estimation method.
    threshold : float
        Significance threshold.
    """

    tag = "ASmoothMapEstimator"

    def __init__(
        self,
        scales=None,
        kernel=Gaussian2DKernel,
        spectrum=None,
        method="lima",
        threshold=5,
        e_edges=None,
    ):
        if spectrum is None:
            spectrum = PowerLawSpectralModel()

        self.spectrum = spectrum

        if scales is None:
            scales = self.get_scales(n_scales=9, kernel=kernel)

        self.scales = scales
        self.kernel = kernel
        self.threshold = threshold
        self.method = method
        self.e_edges = e_edges

    def selection_all(self):
        """Which quantities are computed"""
        return

    @staticmethod
    def get_scales(n_scales, factor=np.sqrt(2), kernel=Gaussian2DKernel):
        """Create list of Gaussian widths.

        Parameters
        ----------
        n_scales : int
            Number of scales
        factor : float
            Incremental factor

        Returns
        -------
        scales : `~numpy.ndarray`
            Scale array
        """
        if kernel == Gaussian2DKernel:
            sigma_0 = 1.0 / np.sqrt(9 * np.pi)
        elif kernel == Tophat2DKernel:
            sigma_0 = 1.0 / np.sqrt(np.pi)

        return sigma_0 * factor ** np.arange(n_scales)

    def get_kernels(self, pixel_scale):
        """Get kernels according to the specified method.

        Parameters
        ----------
        pixel_scale : `~astropy.coordinates.Angle`
            Sky image pixel scale

        Returns
        -------
        kernels : list
            List of `~astropy.convolution.Kernel`
        """
        scales = self.scales.to_value("deg") / Angle(pixel_scale).deg

        kernels = []
        for scale in scales:  # .value:
            kernel = self.kernel(scale, mode="oversample")
            # TODO: check if normalizing here makes sense
            kernel.normalize("peak")
            kernels.append(kernel)

        return kernels

    @staticmethod
    def _sqrt_ts_cube(cubes, method):
        if method in {"lima"}:
            scube = CashCountsStatistic(
                cubes["counts"], cubes["background"]
            ).significance
        elif method == "asmooth":
            scube = _sqrt_ts_asmooth(cubes["counts"], cubes["background"])
        elif method == "ts":
            raise NotImplementedError()
        else:
            raise ValueError(
                "Not a valid sqrt_ts estimation method."
                " Choose one of the following: 'lima' or 'asmooth'"
            )
        return scube

    def run(self, dataset, show_pbar=True):
        """Run adaptive smoothing on input MapDataset.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            the input dataset (with one bin in energy at most)
        show_pbar : bool
            Display progress bar.

        Returns
        -------
        images : dict of `~gammapy.maps.WcsNDMap`
            Smoothed images; keys are:
                * 'counts'
                * 'background'
                * 'flux' (optional)
                * 'scales'
                * 'sqrt_ts'.
        """
        datasets = Datasets([dataset])

        if self.e_edges is None:
            energy_axis = dataset.counts.geom.axes["energy"]
            e_edges = u.Quantity([energy_axis.edges[0], energy_axis.edges[-1]])
        else:
            e_edges = self.e_edges

        results = []

        with pbar(total=len(e_edges) - 1, show_pbar=show_pbar) as pb:
            for e_min, e_max in zip(e_edges[:-1], e_edges[1:]):
                dataset = datasets.slice_energy(e_min, e_max)[0]
                result = self.estimate_maps(dataset)
                results.append(result)
                pb.update(1)

        result_all = {}

        for name in result.keys():
            map_all = Map.from_images(images=[_[name] for _ in results])
            result_all[name] = map_all

        return result_all

    def estimate_maps(self, dataset):
        """
        Run adaptive smoothing on input Maps.

        Parameters
        ----------
        dataset : `MapDataset`
            Dataset

        Returns
        -------
        images : dict of `~gammapy.maps.WcsNDMap`
            Smoothed images; keys are:
                * 'counts'
                * 'background'
                * 'flux' (optional)
                * 'scales'
                * 'sqrt_ts'.
        """
        dataset = dataset.to_image()

        # extract 2d arrays
        counts = dataset.counts.data[0].astype(float)
        background = dataset.background_model.evaluate().data[0]

        if isinstance(dataset, MapDatasetOnOff):
            background = dataset.counts_off_normalised.data[0]

        if dataset.exposure is not None:
            exposure = estimate_exposure_reco_energy(dataset, self.spectrum)
        else:
            exposure = None

        pixel_scale = dataset.counts.geom.pixel_scales.mean()
        kernels = self.get_kernels(pixel_scale)

        cubes = {}
        cubes["counts"] = scale_cube(counts, kernels)
        cubes["background"] = scale_cube(background, kernels)

        if exposure is not None:
            flux = (dataset.counts - background) / exposure
            cubes["flux"] = scale_cube(flux.data[0], kernels)

        cubes["sqrt_ts"] = self._sqrt_ts_cube(cubes, method=self.method)

        smoothed = self._reduce_cubes(cubes, kernels)

        result = {}

        geom = dataset.counts.geom

        for name, data in smoothed.items():
            # set remaining pixels with sqrt_ts < threshold to mean value
            if name in ["counts", "background"]:
                mask = np.isnan(data)
                data[mask] = np.mean(locals()[name][mask])
                result[name] = WcsNDMap(geom, data, unit="")
            else:
                unit = "deg" if name == "scale" else ""
                result[name] = WcsNDMap(geom, data, unit=unit)

        if exposure is not None:
            data = smoothed["flux"]
            mask = np.isnan(data)
            data[mask] = np.mean(flux.data[0][mask])
            result["flux"] = WcsNDMap(geom, data, unit=flux.unit)

        return result

    def _reduce_cubes(self, cubes, kernels):
        """
        Combine scale cube to image.

        Parameters
        ----------
        cubes : dict
            Data cubes
        """
        shape = cubes["counts"].shape[:2]
        smoothed = {}

        # Init smoothed data arrays
        for key in ["counts", "background", "scale", "sqrt_ts"]:
            smoothed[key] = np.tile(np.nan, shape)

        if "flux" in cubes:
            smoothed["flux"] = np.tile(np.nan, shape)

        for idx, scale in enumerate(self.scales):
            # slice out 2D image at index idx out of cube
            slice_ = np.s_[:, :, idx]

            mask = np.isnan(smoothed["counts"])
            mask = (cubes["sqrt_ts"][slice_] > self.threshold) & mask

            smoothed["scale"][mask] = scale
            smoothed["sqrt_ts"][mask] = cubes["sqrt_ts"][slice_][mask]

            # renormalize smoothed data arrays
            norm = kernels[idx].array.sum()
            for key in ["counts", "background"]:
                smoothed[key][mask] = cubes[key][slice_][mask] / norm

            if "flux" in cubes:
                smoothed["flux"][mask] = cubes["flux"][slice_][mask] / norm

        return smoothed

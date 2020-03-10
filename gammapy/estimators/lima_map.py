# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import Angle
from gammapy.maps import Map
from gammapy.datasets import MapDataset, MapDatasetOnOff
from gammapy.stats import significance, significance_on_off
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic

__all__ = [
    "LiMaMapEstimator",
]

log = logging.getLogger(__name__)

def convolved_map_dataset_counts_statistics(dataset, kernel):
    """Return CountsDataset objects containing smoothed maps from the MapDataset"""
    # Kernel is modified later make a copy here
    kernel = copy.deepcopy(kernel)
    kernel.normalize("peak")

    # fft convolution adds numerical noise, to ensure integer results we call
    # np.rint
    n_on_conv = np.rint(dataset.counts.convolve(kernel.array).data)

    if isinstance(dataset, MapDatasetOnOff):
        background = dataset.background
        background.data[dataset.acceptance_off.data == 0] = 0.0
        background_conv = background.convolve(kernel.array).data

        n_off_conv = dataset.counts_off.convolve(kernel.array).data

        with np.errstate(invalid="ignore", divide="ignore"):
            alpha_conv = background_conv / n_off_conv

        return WStatCountsStatistic(n_on_conv.data, n_off_conv.data, alpha_conv.data)
    else:
        background_conv = dataset.npred().convolve(kernel.array).data
        return CashCountsStatistic(n_on_conv.data, background_conv.data)

class LiMaMapEstimator:
    """Computes correlated excess, significance and errors for MapDatasets

    Parameters
    ----------
    dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
        input dataset
    """

    def __init__(self, dataset, nsigma=1, nsigma_ul=3):

        if not isinstance(dataset, MapDataset):
            raise ValueError("Unsupported dataset type")
        self._dataset = dataset

        self.nsigma = nsigma
        self.nsigma_ul = nsigma_ul

    @property
    def dataset(self):
        return self._dataset

    def run(self, correlation_radius, steps="all"):
        """Compute correlated excess, Li & Ma significance and flux maps

        This requires datasets with only 1 energy bins (image-like).
        Usually you can obtain one with `dataset.to_image()`


        Parameters
        ----------
        correlation_radius : ~astropy.coordinate.Angle
            correlation radius to use

        Returns
        -------
        images : dict
            Dictionary containing result maps. Keys are: significance,
            counts, background and excess for a MapDataset significance,
            n_on, background, excess, alpha otherwise

        """
        self.radius = Angle(correlation_radius)

        pixel_size = np.mean(np.abs(self.dataset.counts.geom.wcs.wcs.cdelt))
        size = self.radius.deg / pixel_size
        kernel = Tophat2DKernel(size)

        geom = self.dataset.counts.geom

        counts_stat = convolved_map_dataset_counts_statistics(self.dataset, kernel)

        n_on = Map.from_geom(geom, data=counts_stat.n_on)
        bkg = Map.from_geom(geom, data=counts_stat.n_on-counts_stat.excess)
        excess = Map.from_geom(geom, data=counts_stat.excess)
        significance = Map.from_geom(geom, data=counts_stat.significance)

        result = {"counts": n_on, "background": bkg, "excess": excess, "significance": significance}

        if steps == "all":
            steps = ["err", "errn-errp", "ul"]

        if "err" in steps:
            err = Map.from_geom(geom, data=counts_stat.error)
            result.update({"err": err})

        if "errn-errp" in steps:
            errn = Map.from_geom(geom, data=counts_stat.compute_errn(self.nsigma))
            errp = Map.from_geom(geom, data=counts_stat.compute_errp(self.nsigma))
            result.update({"errn": errn, "errp": errp})

        if "ul" in steps:
            ul = Map.from_geom(geom, data=counts_stat.compute_upper_limit(self.nsigma_ul))
            result.update({"ul": ul})
        return result

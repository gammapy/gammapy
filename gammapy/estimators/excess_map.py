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
    "CorrelatedExcessMapEstimator",
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

class CorrelatedExcessMapEstimator:
    """Computes correlated excess, significance and errors for MapDatasets

    Parameters
    ----------
    dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
        input image-like dataset
    correlation_radius : ~astropy.coordinate.Angle
        correlation radius to use
    n_sigma : float
        Confidence level for the asymmetric errors expressed in number of sigma.
        Default is 1.
    n_sigma_ul : float
        Confidence level for the upper limits expressed in number of sigma.
        Default is 3.
    """

    def __init__(self, dataset, correlation_radius='0.1 deg', nsigma=1, nsigma_ul=3):
        self.dataset = dataset
        self.correlation_radius = correlation_radius
        self.nsigma = nsigma
        self.nsigma_ul = nsigma_ul

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        if not isinstance(dataset, MapDataset):
            raise ValueError("Unsupported dataset type")
        self._dataset = dataset

    @property
    def correlation_radius(self):
        return self._correlation_radius

    @correlation_radius.setter
    def correlation_radius(self, correlation_radius):
        """Sets radius"""
        self._correlation_radius = Angle(correlation_radius)

    def run(self, steps="all"):
        """Compute correlated excess, Li & Ma significance and flux maps

        Parameters
        ----------
        steps : list of str
            Which steps to execute. Available options are:

                * "ts": estimate delta TS and significance
                * "err": estimate symmetric error
                * "errn-errp": estimate asymmetric errors.
                * "ul": estimate upper limits.

            By default all steps are executed.

        Returns
        -------
        images : dict
            Dictionary containing result correlated maps. Keys are:

                * counts : correlated counts map
                * background : correlated background map
                * excess : correlated excess map
                * ts : delta TS map
                * significance : sqrt(delta TS), or Li-Ma significance map
                * err : symmetric error map (from covariance)
                * errn : negative error map
                * errp : positive error map
                * ul : upper limit map

        """
        pixel_size = np.mean(np.abs(self.dataset.counts.geom.wcs.wcs.cdelt))
        size = self.correlation_radius.deg / pixel_size
        kernel = Tophat2DKernel(size)

        geom = self.dataset.counts.geom

        self.counts_stat = convolved_map_dataset_counts_statistics(self.dataset, kernel)

        n_on = Map.from_geom(geom, data=self.counts_stat.n_on)
        bkg = Map.from_geom(geom, data=self.counts_stat.n_on-self.counts_stat.excess)
        excess = Map.from_geom(geom, data=self.counts_stat.excess)

        result = {"counts": n_on, "background": bkg, "excess": excess}

        if steps == "all":
            steps = ["ts", "err", "errn-errp", "ul"]

        if "ts" in steps:
            tsmap = Map.from_geom(geom, data=self.counts_stat.delta_ts)
            significance = Map.from_geom(geom, data=self.counts_stat.significance)
            result.update({"ts": tsmap, "significance": significance})

        if "err" in steps:
            err = Map.from_geom(geom, data=self.counts_stat.error)
            result.update({"err": err})

        if "errn-errp" in steps:
            errn = Map.from_geom(geom, data=self.counts_stat.compute_errn(self.nsigma))
            errp = Map.from_geom(geom, data=self.counts_stat.compute_errp(self.nsigma))
            result.update({"errn": errn, "errp": errp})

        if "ul" in steps:
            ul = Map.from_geom(geom, data=self.counts_stat.compute_upper_limit(self.nsigma_ul))
            result.update({"ul": ul})
        return result

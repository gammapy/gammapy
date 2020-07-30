# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import Angle
from gammapy.datasets import MapDataset, MapDatasetOnOff
from gammapy.maps import Map
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic
from .core import Estimator

__all__ = [
    "ExcessMapEstimator",
]

log = logging.getLogger(__name__)


def convolved_map_dataset_counts_statistics(
    dataset, kernel, apply_mask_fit=False, return_image=False
):
    """Return CountsDataset objects containing smoothed maps from the MapDataset"""
    # Kernel is modified later make a copy here
    kernel = copy.deepcopy(kernel)
    kernel.normalize("peak")

    # fft convolution adds numerical noise, to ensure integer results we call
    # np.rint
    n_on = dataset.counts
    if dataset.mask_safe:
        n_on = n_on * dataset.mask_safe
    if apply_mask_fit:
        n_on = n_on * dataset.mask_fit
    if return_image:
        n_on = n_on.sum_over_axes(keepdims=False)
    n_on_conv = np.rint(n_on.convolve(kernel.array).data)

    if isinstance(dataset, MapDatasetOnOff):
        background = dataset.background
        background.data[dataset.acceptance_off.data == 0] = 0.0
        n_off = dataset.counts_off

        if apply_mask_fit:
            background = background * dataset.mask_fit
            n_off = n_off * dataset.mask_fit
        if return_image:
            background = background.sum_over_axes(keepdims=False)
            n_off = n_off.sum_over_axes(keepdims=False)

        background_conv = background.convolve(kernel.array)
        n_off_conv = n_off.convolve(kernel.array)

        with np.errstate(invalid="ignore", divide="ignore"):
            alpha_conv = background_conv / n_off_conv

        return WStatCountsStatistic(n_on_conv.data, n_off_conv.data, alpha_conv.data)
    else:
        npred = dataset.npred()
        if dataset.mask_safe:
            npred = npred * dataset.mask_safe
        if apply_mask_fit:
            npred = npred * dataset.mask_fit
        if return_image:
            npred = npred.sum_over_axes(keepdims=False)
        background_conv = npred.convolve(kernel.array)
        return CashCountsStatistic(n_on_conv.data, background_conv.data)


class ExcessMapEstimator(Estimator):
    """Computes correlated excess, significance and errors for MapDatasets.

    Parameters
    ----------
    correlation_radius : ~astropy.coordinate.Angle
        correlation radius to use
    n_sigma : float
        Confidence level for the asymmetric errors expressed in number of sigma.
        Default is 1.
    n_sigma_ul : float
        Confidence level for the upper limits expressed in number of sigma.
        Default is 3.
    selection : list of str
        Which additional maps to estimate besides delta TS, significance and symmetric error.
        Available options are:

            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.

        By default all additional quantities are estimated.
    apply_mask_fit : Bool
        Apply a mask for the computation.
        A `~gammapy.datasets.MapDataset.mask_fit` must be present on the input dataset
    return_image : Bool
        Reduce the input dataset to a 2D image and perform the computations.
    """

    tag = "ExcessMapEstimator"
    available_selection = ["errn-errp", "ul"]

    def __init__(self,
        correlation_radius="0.1 deg",
        n_sigma=1,
        n_sigma_ul=3,
        selection='all',
        apply_mask_fit=False,
        return_image=False,
    ):
        self.correlation_radius = correlation_radius
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.apply_mask_fit = apply_mask_fit
        self.return_image = return_image
        self.selection = self._make_selection(selection)

    @property
    def correlation_radius(self):
        return self._correlation_radius

    @correlation_radius.setter
    def correlation_radius(self, correlation_radius):
        """Sets radius"""
        self._correlation_radius = Angle(correlation_radius)

    def run(self, dataset):
        """Compute correlated excess, Li & Ma significance and flux maps

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            input dataset

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
        if not isinstance(dataset, MapDataset):
            raise ValueError("Unsupported dataset type")

        pixel_size = np.mean(np.abs(dataset.counts.geom.wcs.wcs.cdelt))
        size = self.correlation_radius.deg / pixel_size
        kernel = Tophat2DKernel(size)

        geom = dataset.counts.geom

        counts_stat = convolved_map_dataset_counts_statistics(
            dataset, kernel, self.apply_mask_fit, self.return_image
        )

        if self.return_image:
            geom = dataset.counts.geom.to_image()

        n_on = Map.from_geom(geom, data=counts_stat.n_on)
        bkg = Map.from_geom(geom, data=counts_stat.n_on - counts_stat.excess)
        excess = Map.from_geom(geom, data=counts_stat.excess)

        result = {"counts": n_on, "background": bkg, "excess": excess}

        tsmap = Map.from_geom(geom, data=counts_stat.delta_ts)
        significance = Map.from_geom(geom, data=counts_stat.significance)
        result.update({"ts": tsmap, "significance": significance})

        err = Map.from_geom(geom, data=counts_stat.error * self.n_sigma)
        result.update({"err": err})

        if "errn-errp" in self.selection:
            errn = Map.from_geom(geom, data=counts_stat.compute_errn(self.n_sigma))
            errp = Map.from_geom(geom, data=counts_stat.compute_errp(self.n_sigma))
            result.update({"errn": errn, "errp": errp})

        if "ul" in self.selection:
            ul = Map.from_geom(
                geom, data=counts_stat.compute_upper_limit(self.n_sigma_ul)
            )
            result.update({"ul": ul})
        return result

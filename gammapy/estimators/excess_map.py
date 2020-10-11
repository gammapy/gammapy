# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import Angle
import astropy.units as u
from gammapy.utils.pbar import pbar
from gammapy.datasets import MapDataset, MapDatasetOnOff, Datasets
from gammapy.maps import Map
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic
from .core import Estimator
from .utils import estimate_exposure_reco_energy

__all__ = [
    "ExcessMapEstimator",
]

log = logging.getLogger(__name__)


def convolved_map_dataset_counts_statistics(dataset, kernel, apply_mask_fit=False):
    """Return CountsDataset objects containing smoothed maps from the MapDataset"""
    # Kernel is modified later make a copy here
    kernel = copy.deepcopy(kernel)
    kernel.normalize("peak")

    mask = np.ones(dataset.data_shape, dtype=bool)
    if dataset.mask_safe:
        mask *= dataset.mask_safe
    if apply_mask_fit:
        mask *= dataset.mask_fit

    # fft convolution adds numerical noise, to ensure integer results we call
    # np.rint
    n_on = dataset.counts * mask
    n_on = n_on.sum_over_axes(keepdims=True)
    n_on_conv = np.rint(n_on.convolve(kernel.array).data)

    if isinstance(dataset, MapDatasetOnOff):
        background = dataset.counts_off_normalised * mask
        background.data[dataset.acceptance_off.data == 0] = 0.0
        n_off = dataset.counts_off * mask

        background = background.sum_over_axes(keepdims=True)
        n_off = n_off.sum_over_axes(keepdims=True)

        background_conv = background.convolve(kernel.array)
        n_off_conv = n_off.convolve(kernel.array)

        npred_sig = dataset.npred_sig() * mask
        npred_sig = npred_sig.sum_over_axes(keepdims=True)
        mu_sig = npred_sig.convolve(kernel.array)

        with np.errstate(invalid="ignore", divide="ignore"):
            alpha_conv = background_conv / n_off_conv

        return WStatCountsStatistic(
            n_on_conv.data, n_off_conv.data, alpha_conv.data, mu_sig.data
        )
    else:

        npred = dataset.npred() * mask
        npred = npred.sum_over_axes(keepdims=True)
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
    selection_optional : list of str
        Which additional maps to estimate besides delta TS, significance and symmetric error.
        Available options are:

            * "flux": estimate flux map
            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.

        By default all additional quantities are estimated.
    e_edges : `~astropy.units.Quantity`
        Energy edges of the target excess maps bins.
    apply_mask_fit : Bool
        Apply a mask for the computation.
        A `~gammapy.datasets.MapDataset.mask_fit` must be present on the input dataset
    """

    tag = "ExcessMapEstimator"
    _available_selection_optional = ["errn-errp", "ul"]

    def __init__(
        self,
        correlation_radius="0.1 deg",
        n_sigma=1,
        n_sigma_ul=3,
        selection_optional="all",
        e_edges=None,
        apply_mask_fit=False,
        return_image=False,
    ):
        self.correlation_radius = correlation_radius
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.apply_mask_fit = apply_mask_fit
        self.selection_optional = selection_optional
        self.e_edges = e_edges

    @property
    def correlation_radius(self):
        return self._correlation_radius

    @correlation_radius.setter
    def correlation_radius(self, correlation_radius):
        """Sets radius"""
        self._correlation_radius = Angle(correlation_radius)

    def run(self, dataset, show_pbar=True):
        """Compute correlated excess, Li & Ma significance and flux maps

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            input dataset
        show_pbar : bool
            Display progress bar.

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
                * flux : flux map. An exposure map must be present in the dataset to compute flux map
                * errn : negative error map
                * errp : positive error map
                * ul : upper limit map

        """
        if not isinstance(dataset, MapDataset):
            raise ValueError("Unsupported dataset type")

        # TODO: add support for joint excess estimate to ExcessMapEstimator?
        datasets = Datasets(dataset)

        if self.e_edges is None:
            energy_axis = dataset.counts.geom.axes["energy"]
            e_edges = u.Quantity([energy_axis.edges[0], energy_axis.edges[-1]])
        else:
            e_edges = self.e_edges

        results = []

        with pbar(total=len(e_edges) - 1, show_pbar=show_pbar) as pb:
            for e_min, e_max in zip(e_edges[:-1], e_edges[1:]):
                sliced_dataset = datasets.slice_energy(e_min, e_max)[0]

                result = self.estimate_excess_map(sliced_dataset)
                results.append(result)
                pb.update(1)

        results_all = {}

        for name in results[0].keys():
            map_all = Map.from_images(images=[_[name] for _ in results])
            results_all[name] = map_all

        return results_all

    def estimate_excess_map(self, dataset):
        """Estimate excess and ts maps for single dataset.

        If exposure is defined, a flux map is also computed.

        Parameters
        ----------
        dataset : `MapDataset`
            Map dataset
        """

        pixel_size = np.mean(np.abs(dataset.counts.geom.wcs.wcs.cdelt))
        size = self.correlation_radius.deg / pixel_size
        kernel = Tophat2DKernel(size)

        counts_stat = convolved_map_dataset_counts_statistics(
            dataset, kernel, self.apply_mask_fit
        )

        geom = dataset.counts.geom.squash("energy")

        n_on = Map.from_geom(geom, data=counts_stat.n_on)
        bkg = Map.from_geom(geom, data=counts_stat.n_on - counts_stat.excess)
        excess = Map.from_geom(geom, data=counts_stat.excess)

        result = {"counts": n_on, "background": bkg, "excess": excess}

        tsmap = Map.from_geom(geom, data=counts_stat.delta_ts)
        significance = Map.from_geom(geom, data=counts_stat.significance)
        result.update({"ts": tsmap, "significance": significance})

        err = Map.from_geom(geom, data=counts_stat.error * self.n_sigma)
        result.update({"err": err})

        if dataset.exposure:
            reco_exposure = estimate_exposure_reco_energy(dataset)
            reco_exposure = reco_exposure.sum_over_axes(keepdims=True)
            flux = excess / reco_exposure
            flux.quantity = flux.quantity.to("1 / (cm2 s)")
        else:
            flux = Map.from_geom(
                geom=dataset.counts.geom, data=np.nan * np.ones(dataset.data_shape)
            )
        result.update({"flux": flux})

        if "errn-errp" in self.selection_optional:
            errn = Map.from_geom(geom, data=counts_stat.compute_errn(self.n_sigma))
            errp = Map.from_geom(geom, data=counts_stat.compute_errp(self.n_sigma))
            result.update({"errn": errn, "errp": errp})

        if "ul" in self.selection_optional:
            ul = Map.from_geom(
                geom, data=counts_stat.compute_upper_limit(self.n_sigma_ul)
            )
            result.update({"ul": ul})

        return result

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import Angle
from gammapy.datasets import MapDataset, MapDatasetOnOff
from gammapy.maps import Map, MapAxis
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic
from .core import Estimator
from .utils import estimate_exposure_reco_energy

__all__ = [
    "ExcessMapEstimator",
]

log = logging.getLogger(__name__)


def convolved_map_dataset_counts_statistics(dataset, kernel, mask):
    """Return CountsDataset objects containing smoothed maps from the MapDataset"""
    # Kernel is modified later make a copy here
    kernel = copy.deepcopy(kernel)
    kernel.normalize("peak")

    # fft convolution adds numerical noise, to ensure integer results we call
    # np.rint
    n_on = dataset.counts * mask
    n_on_conv = np.rint(n_on.convolve(kernel.array).data)

    if isinstance(dataset, MapDatasetOnOff):
        background = dataset.background * mask
        background.data[dataset.acceptance_off.data == 0] = 0.0
        n_off = dataset.counts_off * mask

        background_conv = background.convolve(kernel.array)
        n_off_conv = n_off.convolve(kernel.array)

        npred_sig = dataset.npred_signal() * mask
        mu_sig = npred_sig.convolve(kernel.array)

        with np.errstate(invalid="ignore", divide="ignore"):
            alpha_conv = background_conv / n_off_conv

        return WStatCountsStatistic(
            n_on_conv.data, n_off_conv.data, alpha_conv.data, mu_sig.data
        )
    else:

        npred = dataset.npred() * mask
        background_conv = npred.convolve(kernel.array)
        return CashCountsStatistic(n_on_conv.data, background_conv.data)


class ExcessMapEstimator(Estimator):
    """Computes correlated excess, sqrt TS (i.e. Li-Ma significance) and errors for MapDatasets.

    If a model is set on the dataset the excess map estimator will compute the excess taking into account
    the predicted counts of the model.

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

            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.

        By default all additional quantities are estimated.
    energy_edges : `~astropy.units.Quantity`
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
        energy_edges=None,
        apply_mask_fit=False,
    ):
        self.correlation_radius = correlation_radius
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.apply_mask_fit = apply_mask_fit
        self.selection_optional = selection_optional
        self.energy_edges = energy_edges

    @property
    def correlation_radius(self):
        return self._correlation_radius

    @correlation_radius.setter
    def correlation_radius(self, correlation_radius):
        """Sets radius"""
        self._correlation_radius = Angle(correlation_radius)

    def run(self, dataset):
        """Compute correlated excess, Li & Ma significance and flux maps

        If a model is set on the dataset the excess map estimator will compute the excess taking into account
        the predicted counts of the model.

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
                * ts : TS map
                * sqrt_ts : sqrt(delta TS), or Li-Ma significance map
                * err : symmetric error map (from covariance)
                * flux : flux map. An exposure map must be present in the dataset to compute flux map
                * errn : negative error map
                * errp : positive error map
                * ul : upper limit map

        """
        if not isinstance(dataset, MapDataset):
            raise ValueError("Unsupported dataset type")

        if self.energy_edges is None:
            energy_axis = dataset.counts.geom.axes["energy"]
            energy_edges = u.Quantity([energy_axis.edges[0], energy_axis.edges[-1]])
        else:
            energy_edges = self.energy_edges

        axis = MapAxis.from_energy_edges(energy_edges)

        resampled_dataset = dataset.resample_energy_axis(energy_axis=axis)

        # Beware we rely here on the correct npred background in MapDataset.resample_energy_axis
        resampled_dataset.models = dataset.models

        result = self.estimate_excess_map(resampled_dataset)

        return result

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

        geom = dataset.counts.geom

        if self.apply_mask_fit:
            mask = dataset.mask
        elif dataset.mask_safe:
            mask = dataset.mask_safe
        else:
            mask = np.ones(dataset.data_shape, dtype=bool)

        counts_stat = convolved_map_dataset_counts_statistics(dataset, kernel, mask)

        n_on = Map.from_geom(geom, data=counts_stat.n_on)
        bkg = Map.from_geom(geom, data=counts_stat.n_on - counts_stat.n_sig)
        excess = Map.from_geom(geom, data=counts_stat.n_sig)

        result = {"counts": n_on, "background": bkg, "excess": excess}

        tsmap = Map.from_geom(geom, data=counts_stat.ts)
        sqrt_ts = Map.from_geom(geom, data=counts_stat.sqrt_ts)
        result.update({"ts": tsmap, "sqrt_ts": sqrt_ts})

        err = Map.from_geom(geom, data=counts_stat.error * self.n_sigma)
        result.update({"err": err})

        if dataset.exposure:
            reco_exposure = estimate_exposure_reco_energy(dataset)
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

        # return nan values outside mask
        for key in result:
            result[key].data[~mask] = np.nan

        return result

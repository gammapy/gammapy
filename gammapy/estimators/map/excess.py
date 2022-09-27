# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import Angle
from gammapy.datasets import MapDataset, MapDatasetOnOff
from gammapy.maps import Map
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic
from ..core import Estimator
from ..utils import estimate_exposure_reco_energy
from .core import FluxMaps

__all__ = [
    "ExcessMapEstimator",
]

log = logging.getLogger(__name__)


def convolved_map_dataset_counts_statistics(dataset, kernel, mask, correlate_off):
    """Return CountsDataset objects containing smoothed maps from the MapDataset"""
    # Kernel is modified later make a copy here
    kernel = copy.deepcopy(kernel)
    kernel.normalize("peak")

    # fft convolution adds numerical noise, to ensure integer results we call
    # np.rint
    n_on = dataset.counts * mask
    n_on_conv = np.rint(n_on.convolve(kernel.array).data)

    if isinstance(dataset, MapDatasetOnOff):
        n_off = dataset.counts_off * mask
        npred_sig = dataset.npred_signal() * mask
        acceptance_on = dataset.acceptance * mask
        acceptance_off = dataset.acceptance_off * mask

        npred_sig_convolve = npred_sig.convolve(kernel.array)
        if correlate_off:
            background = dataset.background * mask
            background.data[dataset.acceptance_off == 0] = 0.0
            background_conv = background.convolve(kernel.array)

            n_off = n_off.convolve(kernel.array)
            with np.errstate(invalid="ignore", divide="ignore"):
                alpha = background_conv / n_off

        else:
            acceptance_on_convolve = acceptance_on.convolve(kernel.array)

            with np.errstate(invalid="ignore", divide="ignore"):
                alpha = acceptance_on_convolve / acceptance_off

        return WStatCountsStatistic(
            n_on_conv.data, n_off.data, alpha.data, npred_sig_convolve.data
        )
    else:

        npred = dataset.npred() * mask
        background_conv = npred.convolve(kernel.array)
        return CashCountsStatistic(n_on_conv.data, background_conv.data)


class ExcessMapEstimator(Estimator):
    """Computes correlated excess, significance and error maps from a map dataset.

    If a model is set on the dataset the excess map estimator will compute the
    excess taking into account the predicted counts of the model.

    ..note::

        By default the excess estimator correlates the off counts as well to avoid
        artifacts at the edges of the :term:`FoV` for stacked on-off datasets.
        However when the on-off dataset has been derived from a ring background
        estimate, this leads to the off counts being correlated twice. To avoid
        artifacts and the double correlation, the `ExcessMapEstimator` has to
        be applied per dataset and the resulting maps need to be stacked, taking
        the :term:`FoV` cut into account.

    Parameters
    ----------
    correlation_radius : ~astropy.coordinate.Angle
        correlation radius to use
    n_sigma : float
        Confidence level for the asymmetric errors expressed in number of sigma.
    n_sigma_ul : float
        Confidence level for the upper limits expressed in number of sigma.
    selection_optional : list of str
        Which additional maps to estimate besides delta TS, significance and symmetric error.
        Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.

        Default is None so the optional steps are not executed.
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the target excess maps bins.
    correlate_off : bool
        Correlate OFF events. Default is True.
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral model used for the computation of the flux map.
        If None, a Power Law of index 2 is assumed (default).

    Examples
    --------
    >>> from gammapy.datasets import MapDataset
    >>> from gammapy.estimators import ExcessMapEstimator
    >>> dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
    >>> estimator = ExcessMapEstimator(correlation_radius="0.1 deg")
    >>> result = estimator.run(dataset)
    >>> print(result)
    FluxMaps
    --------
    <BLANKLINE>
      geom                   : WcsGeom
      axes                   : ['lon', 'lat', 'energy']
      shape                  : (320, 240, 1)
      quantities             : ['npred', 'npred_excess', 'counts', 'ts', 'sqrt_ts', 'norm', 'norm_err']  # noqa: E501
      ref. model             : pl
      n_sigma                : 1
      n_sigma_ul             : 2
      sqrt_ts_threshold_ul   : 2
      sed type init          : likelihood

    """

    tag = "ExcessMapEstimator"
    _available_selection_optional = ["errn-errp", "ul"]

    def __init__(
        self,
        correlation_radius="0.1 deg",
        n_sigma=1,
        n_sigma_ul=2,
        selection_optional=None,
        energy_edges=None,
        correlate_off=True,
        spectral_model=None,
    ):
        self.correlation_radius = correlation_radius
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.selection_optional = selection_optional
        self.energy_edges = energy_edges
        self.correlate_off = correlate_off

        if spectral_model is None:
            spectral_model = PowerLawSpectralModel(index=2)

        self.spectral_model = spectral_model

    @property
    def correlation_radius(self):
        return self._correlation_radius

    @correlation_radius.setter
    def correlation_radius(self, correlation_radius):
        """Sets radius"""
        self._correlation_radius = Angle(correlation_radius)

    def run(self, dataset):
        """Compute correlated excess, Li & Ma significance and flux maps

        If a model is set on the dataset the excess map estimator will compute
        the excess taking into account the predicted counts of the model.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            Map dataset

        Returns
        -------
        maps : `FluxMaps`
            Flux maps
        """
        if not isinstance(dataset, MapDataset):
            raise ValueError(
                "Unsupported dataset type. Excess map is not applicable to 1D datasets."
            )

        axis = self._get_energy_axis(dataset)

        resampled_dataset = dataset.resample_energy_axis(
            energy_axis=axis, name=dataset.name
        )
        if isinstance(dataset, MapDatasetOnOff):
            resampled_dataset.models = dataset.models
        else:
            resampled_dataset.background = dataset.npred().resample_axis(axis=axis)
            resampled_dataset.models = None

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

        if dataset.mask_fit:
            mask = dataset.mask
        elif dataset.mask_safe:
            mask = dataset.mask_safe
        else:
            mask = Map.from_geom(geom, data=True, dtype=bool)

        counts_stat = convolved_map_dataset_counts_statistics(
            dataset, kernel, mask, self.correlate_off
        )

        maps = {}
        maps["npred"] = Map.from_geom(geom, data=counts_stat.n_on)
        maps["npred_excess"] = Map.from_geom(geom, data=counts_stat.n_sig)
        maps["counts"] = maps["npred"]

        maps["ts"] = Map.from_geom(geom, data=counts_stat.ts)
        maps["sqrt_ts"] = Map.from_geom(geom, data=counts_stat.sqrt_ts)

        if dataset.exposure:
            reco_exposure = estimate_exposure_reco_energy(
                dataset, self.spectral_model, normalize=False
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                reco_exposure = reco_exposure.convolve(kernel.array) / mask.convolve(
                    kernel.array
                )
        else:
            reco_exposure = 1

        with np.errstate(invalid="ignore", divide="ignore"):
            maps["norm"] = maps["npred_excess"] / reco_exposure
            maps["norm_err"] = (
                Map.from_geom(geom, data=counts_stat.error * self.n_sigma)
                / reco_exposure
            )

            if "errn-errp" in self.selection_optional:
                maps["norm_errn"] = (
                    Map.from_geom(geom, data=counts_stat.compute_errn(self.n_sigma))
                    / reco_exposure
                )
                maps["norm_errp"] = (
                    Map.from_geom(geom, data=counts_stat.compute_errp(self.n_sigma))
                    / reco_exposure
                )

            if "ul" in self.selection_optional:
                maps["norm_ul"] = (
                    Map.from_geom(
                        geom, data=counts_stat.compute_upper_limit(self.n_sigma_ul)
                    )
                    / reco_exposure
                )

        # return nan values outside mask
        for name in maps:
            maps[name].data[~mask] = np.nan

        meta = {
            "n_sigma": self.n_sigma,
            "n_sigma_ul": self.n_sigma_ul,
            "sed_type_init": "likelihood",
        }

        return FluxMaps.from_maps(
            maps=maps,
            meta=meta,
            reference_model=SkyModel(self.spectral_model),
            sed_type="likelihood",
        )

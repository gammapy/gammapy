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

__all__ = ["ExcessMapEstimator"]

log = logging.getLogger(__name__)


def _get_convolved_maps(dataset, kernel, mask, correlate_off):
    """Return convolved maps.

    Parameters
    ----------
    dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
        Map dataset.
    kernel : `~astropy.convolution.Kernel`
        Kernel.
    mask : `~gammapy.maps.Map`
        Mask map.
    correlate_off : bool
        Correlate OFF events.

    Returns
    -------
    convolved_maps : dict
        Dictionary of convolved maps.
    """
    # Kernel is modified later make a copy here
    kernel = copy.deepcopy(kernel)
    kernel_data = kernel.data / kernel.data.max()

    # fft convolution adds numerical noise, to ensure integer results we call
    # np.rint
    n_on = dataset.counts * mask
    n_on_conv = np.rint(n_on.convolve(kernel_data).data)

    convolved_maps = {"n_on_conv": n_on_conv}

    if isinstance(dataset, MapDatasetOnOff):
        n_off = dataset.counts_off * mask
        npred_sig = dataset.npred_signal() * mask
        acceptance_on = dataset.acceptance * mask
        acceptance_off = dataset.acceptance_off * mask
        npred_sig_convolve = npred_sig.convolve(kernel_data)
        if correlate_off:
            background = dataset.background * mask
            background.data[dataset.acceptance_off == 0] = 0.0
            background_conv = background.convolve(kernel_data)
            n_off = n_off.convolve(kernel_data)

            with np.errstate(invalid="ignore", divide="ignore"):
                alpha = background_conv / n_off

        else:
            acceptance_on_convolve = acceptance_on.convolve(kernel_data)

            with np.errstate(invalid="ignore", divide="ignore"):
                alpha = acceptance_on_convolve / acceptance_off
        convolved_maps.update(
            {
                "n_off": n_off,
                "npred_sig_convolve": npred_sig_convolve,
                "acceptance_on": acceptance_on,
                "acceptance_off": acceptance_off,
                "alpha": alpha,
            }
        )
    else:
        npred = dataset.npred() * mask
        background_conv = npred.convolve(kernel_data)
        convolved_maps.update(
            {
                "background_conv": background_conv,
            }
        )

    return convolved_maps


def convolved_map_dataset_counts_statistics(convolved_maps, stat_type):
    """Return a `CountsStatistic` object.

    Parameters
    ----------
    convolved_maps : dict
        Dictionary of convolved maps.
    stat_type : str
        The statistic type, either 'wstat' or 'cash'.

    Returns
    -------
    counts_statistic : `~gammapy.stats.CashCountsStatistic` or `~gammapy.stats.WStatCountsStatistic`
        The counts statistic.
    """
    if stat_type == "wstat":
        n_on_conv = convolved_maps["n_on_conv"]
        n_off = convolved_maps["n_off"]
        alpha = convolved_maps["alpha"]
        npred_sig_convolve = convolved_maps["npred_sig_convolve"]

        return WStatCountsStatistic(
            n_on_conv.data, n_off.data, alpha.data, npred_sig_convolve.data
        )
    elif stat_type == "cash":
        n_on_conv = convolved_maps["n_on_conv"]
        background_conv = convolved_maps["background_conv"]
        return CashCountsStatistic(n_on_conv.data, background_conv.data)


class ExcessMapEstimator(Estimator):
    """Computes correlated excess, significance and error maps from a map dataset.

    If a model is set on the dataset the excess map estimator will compute the
    excess taking into account the predicted counts of the model.

    .. note::

        By default, the excess estimator correlates the off counts as well to avoid
        artifacts at the edges of the :term:`FoV` for stacked on-off datasets.
        However, when the on-off dataset has been derived from a ring background
        estimate, this leads to the off counts being correlated twice. To avoid
        artifacts and the double correlation, the `ExcessMapEstimator` has to
        be applied per dataset and the resulting maps need to be stacked, taking
        the :term:`FoV` cut into account.

    Parameters
    ----------
    correlation_radius : `~astropy.coordinates.Angle`
        Correlation radius to use.
    n_sigma : float
        Confidence level for the asymmetric errors expressed in number of sigma.
    n_sigma_ul : float
        Confidence level for the upper limits expressed in number of sigma.
    n_sigma_sensitivity : float
        Confidence level for the sensitivity expressed in number of sigma.
    gamma_min_sensitivity : float, optional
        Minimum number of gamma-rays. Default is 10.
    bkg_syst_fraction_sensitivity : float, optional
        Fraction of background counts that are above the gamma-ray counts. Default is 0.05.
    apply_threshold_sensitivity : bool
        If True, use `bkg_syst_fraction_sensitivity` and `gamma_min_sensitivity` in the sensitivity computation.
        Default is False which is the same setting as the HGPS catalog.
    selection_optional : list of str, optional
        Which additional maps to estimate besides delta TS, significance and symmetric error.
        Available options are:

            * "all": all the optional steps are executed.
            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.
            * "sensitivity": estimate sensitivity for a given significance.
            * "alpha": normalisation factor to accounts for differences between the on and off regions.
            * "acceptance_on": acceptance from the on region.
            * "acceptance_off": acceptange from the off region.

        Default is None so the optional steps are not executed.
        Note: "alpha", "acceptance_on" and "acceptance_off" can only be selected if the dataset is a
        `~gammapy.datasets.MapDatasetOnOff`.
    energy_edges : list of `~astropy.units.Quantity`, optional
        Edges of the target maps energy bins. The resulting bin edges won't be exactly equal to the input ones,
        but rather the closest values to the energy axis edges of the parent dataset.
        Default is None: apply the estimator in each energy bin of the parent dataset.
        For further explanation see :ref:`estimators`.
    correlate_off : bool
        Correlate OFF events. Default is True.
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral model used for the computation of the flux map.
        If None, a `~gammapy.modeling.models.PowerLawSpectralModel` of index 2 is assumed (default).
    sum_over_energy_groups : bool
        Only used if ``energy_edges`` is None.
        If False, apply the estimator in each energy bin of the parent dataset.
        If True, apply the estimator in only one bin defined by the energy edges of the parent dataset.
        Default is False.

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
      shape                  : (np.int64(320), np.int64(240), 1)
      quantities             : ['npred', 'npred_excess', 'counts', 'ts', 'sqrt_ts', 'norm', 'norm_err']
      ref. model             : pl
      n_sigma                : 1
      n_sigma_ul             : 2
      sqrt_ts_threshold_ul   : 2
      sed type init          : likelihood

    """

    tag = "ExcessMapEstimator"
    _available_selection_optional = [
        "errn-errp",
        "ul",
        "sensitivity",
        "alpha",
        "acceptance_on",
        "acceptance_off",
    ]

    def __init__(
        self,
        correlation_radius="0.1 deg",
        n_sigma=1,
        n_sigma_ul=2,
        selection_optional=None,
        energy_edges=None,
        correlate_off=True,
        spectral_model=None,
        n_sigma_sensitivity=5,
        gamma_min_sensitivity=10,
        bkg_syst_fraction_sensitivity=0.05,
        apply_threshold_sensitivity=False,
        sum_over_energy_groups=False,
    ):
        self.correlation_radius = correlation_radius
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.n_sigma_sensitivity = n_sigma_sensitivity
        self.gamma_min_sensitivity = gamma_min_sensitivity
        self.bkg_syst_fraction_sensitivity = bkg_syst_fraction_sensitivity
        self.apply_threshold_sensitivity = apply_threshold_sensitivity
        self.selection_optional = selection_optional
        self.energy_edges = energy_edges
        self.sum_over_energy_groups = sum_over_energy_groups
        self.correlate_off = correlate_off

        if spectral_model is None:
            spectral_model = PowerLawSpectralModel(index=2)

        self.spectral_model = spectral_model

    @property
    def correlation_radius(self):
        return self._correlation_radius

    @correlation_radius.setter
    def correlation_radius(self, correlation_radius):
        """Set radius."""
        self._correlation_radius = Angle(correlation_radius)

    def run(self, dataset):
        """Compute correlated excess, Li & Ma significance and flux maps.

        If a model is set on the dataset the excess map estimator will compute
        the excess taking into account the predicted counts of the model.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            Map dataset.

        Returns
        -------
        maps : `FluxMaps`
            Flux maps.
        """
        if not isinstance(dataset, MapDataset):
            raise ValueError(
                "Unsupported dataset type. Excess map is not applicable to 1D datasets."
            )

        axis = self._get_energy_axis(dataset)

        resampled_dataset = dataset.resample_energy_axis(
            energy_axis=axis, name=dataset.name
        )

        if dataset.exposure:
            reco_exposure = estimate_exposure_reco_energy(
                dataset, self.spectral_model, normalize=False
            )
            reco_exposure = reco_exposure.resample_axis(
                axis=axis, weights=dataset.mask_safe
            )
        else:
            reco_exposure = None

        if isinstance(dataset, MapDatasetOnOff):
            resampled_dataset.models = dataset.models
        else:
            resampled_dataset.background = dataset.npred().resample_axis(
                axis=axis, weights=dataset.mask_safe
            )
            resampled_dataset.models = None

        result = self.estimate_excess_map(resampled_dataset, reco_exposure)
        return result

    def estimate_kernel(self, dataset):
        """Get the convolution kernel for the input dataset.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.

        Returns
        -------
        kernel : `~astropy.convolution.Tophat2DKernel`
            Kernel.
        """
        pixel_size = np.mean(np.abs(dataset.counts.geom.wcs.wcs.cdelt))
        size = self.correlation_radius.deg / pixel_size
        kernel = Tophat2DKernel(size)

        geom = dataset.counts.geom.to_image()
        geom = geom.to_odd_npix(max_radius=self.correlation_radius)
        return Map.from_geom(geom, data=kernel.array)

    @staticmethod
    def estimate_mask_default(dataset):
        """Get mask used by the estimator.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.

        Returns
        -------
        mask : `~gammapy.maps.Map`
            Mask map.
        """
        if dataset.mask_fit:
            mask = dataset.mask
        elif dataset.mask_safe:
            mask = dataset.mask_safe
        else:
            mask = Map.from_geom(dataset.counts.geom, data=True, dtype=bool)
        return mask

    def estimate_exposure_reco_energy(self, dataset, kernel, mask, reco_exposure):
        """Estimate exposure map in reconstructed energy for a single dataset assuming the given spectral_model shape.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset.
        kernel : `~astropy.convolution.Tophat2DKernel`
            Kernel.
        mask : `~gammapy.maps.Map`
            Mask map.

        Returns
        -------
        reco_exposure : `~gammapy.maps.Map`
            Reconstructed exposure map.
        """
        if dataset.exposure:
            with np.errstate(invalid="ignore", divide="ignore"):
                reco_exposure = reco_exposure.convolve(kernel.data) / mask.convolve(
                    kernel.data
                )
        else:
            reco_exposure = 1

        return reco_exposure

    def estimate_excess_map(self, dataset, reco_exposure):
        """Estimate excess and test statistic maps for a single dataset.

        If exposure is defined, a flux map is also computed.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset.
        """
        kernel = self.estimate_kernel(dataset)
        geom = dataset.counts.geom
        mask = self.estimate_mask_default(dataset)

        convolved_maps = _get_convolved_maps(dataset, kernel, mask, self.correlate_off)
        counts_stat = convolved_map_dataset_counts_statistics(
            convolved_maps=convolved_maps, stat_type=dataset.stat_type
        )

        maps = {}
        maps["npred"] = Map.from_geom(geom, data=counts_stat.n_on)
        maps["npred_excess"] = Map.from_geom(geom, data=counts_stat.n_sig)
        maps["counts"] = maps["npred"]

        maps["ts"] = Map.from_geom(geom, data=counts_stat.ts)
        maps["sqrt_ts"] = Map.from_geom(geom, data=counts_stat.sqrt_ts)

        reco_exposure = self.estimate_exposure_reco_energy(
            dataset, kernel, mask, reco_exposure
        )

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
            if "sensitivity" in self.selection_optional:
                excess_counts = counts_stat.n_sig_matching_significance(
                    self.n_sigma_sensitivity
                )
                if self.apply_threshold_sensitivity:
                    is_gamma_limited = excess_counts < self.gamma_min_sensitivity
                    excess_counts[is_gamma_limited] = self.gamma_min_sensitivity
                    bkg_syst_limited = (
                        excess_counts
                        < self.bkg_syst_fraction_sensitivity * dataset.background.data
                    )
                    excess_counts[bkg_syst_limited] = (
                        self.bkg_syst_fraction_sensitivity
                        * dataset.background.data[bkg_syst_limited]
                    )
                excess = Map.from_geom(geom=geom, data=excess_counts)
                maps["norm_sensitivity"] = excess / reco_exposure
            if isinstance(dataset, MapDatasetOnOff):
                keys_onoff = set(["alpha", "acceptance_on", "acceptance_off"])
                for key in keys_onoff.intersection(self.selection_optional):
                    maps[key] = convolved_maps[key]

        # return nan values outside mask
        for name in maps:
            maps[name].data[~mask] = np.nan

        meta = {
            "n_sigma": self.n_sigma,
            "n_sigma_ul": self.n_sigma_ul,
            "n_sigma_sensitivity": self.n_sigma_sensitivity,
            "sed_type_init": "likelihood",
        }
        if self.apply_threshold_sensitivity:
            meta["gamma_min_sensitivity"] = self.gamma_min_sensitivity
            meta["bkg_syst_fraction_sensitivity"] = self.bkg_syst_fraction_sensitivity

        return FluxMaps.from_maps(
            maps=maps,
            meta=meta,
            reference_model=SkyModel(self.spectral_model),
            sed_type="likelihood",
        )

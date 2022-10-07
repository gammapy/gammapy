# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from gammapy.datasets import Datasets
from gammapy.maps import MapAxis
from gammapy.modeling import Fit
from gammapy.utils.pbar import progress_bar
from gammapy.utils.table import table_from_row_data
from ..flux import FluxEstimator
from .core import FluxPoints

log = logging.getLogger(__name__)

__all__ = ["FluxPointsEstimator"]


class FluxPointsEstimator(FluxEstimator):
    """Flux points estimator.

    Estimates flux points for a given list of datasets, energies and spectral model.

    To estimate the flux point the amplitude of the reference spectral model is
    fitted within the energy range defined by the energy group. This is done for
    each group independently. The amplitude is re-normalized using the "norm" parameter,
    which specifies the deviation of the flux from the reference model in this
    energy group. See https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/binned_likelihoods/index.html  # noqa: E501
    for details.

    The method is also described in the Fermi-LAT catalog paper
    https://ui.adsabs.harvard.edu/abs/2015ApJS..218...23A
    or the HESS Galactic Plane Survey paper
    https://ui.adsabs.harvard.edu/abs/2018A%26A...612A...1H

    Parameters
    ----------
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the flux point bins.
    source : str or int
        For which source in the model to compute the flux points.
    norm_min : float
        Minimum value for the norm used for the fit statistic profile evaluation.
    norm_max : float
        Maximum value for the norm used for the fit statistic profile evaluation.
    norm_n_values : int
        Number of norm values used for the fit statistic profile.
    norm_values : `numpy.ndarray`
        Array of norm values to be used for the fit statistic profile.
    n_sigma : int
        Number of sigma to use for asymmetric error computation. Default is 1.
    n_sigma_ul : int
        Number of sigma to use for upper limit computation. Default is 2.
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors on flux.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optional steps are not executed.
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.
    sum_over_energy_groups : bool
        Whether to sum over the energy groups or fit the norm on the full energy
        grid.
    """

    tag = "FluxPointsEstimator"

    def __init__(
        self, energy_edges=[1, 10] * u.TeV, sum_over_energy_groups=False, **kwargs
    ):
        self.energy_edges = energy_edges
        self.sum_over_energy_groups = sum_over_energy_groups

        fit = Fit(confidence_opts={"backend": "scipy"})
        kwargs.setdefault("fit", fit)
        super().__init__(**kwargs)

    def run(self, datasets):
        """Run the flux point estimator for all energy groups.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets

        Returns
        -------
        flux_points : `FluxPoints`
            Estimated flux points.
        """
        datasets = Datasets(datasets=datasets)

        if not datasets.energy_axes_are_aligned:
            raise ValueError("All datasets must have aligned energy axes.")

        if "TELESCOP" in datasets.meta_table.colnames:
            telescopes = datasets.meta_table["TELESCOP"]
            if not len(np.unique(telescopes)) == 1:
                raise ValueError(
                    "All datasets must use the same value of the"
                    " 'TELESCOP' meta keyword."
                )

        rows = []
        for energy_min, energy_max in progress_bar(
            zip(self.energy_edges[:-1], self.energy_edges[1:]), desc="Energy bins"
        ):
            row = self.estimate_flux_point(
                datasets,
                energy_min=energy_min,
                energy_max=energy_max,
            )
            rows.append(row)

        meta = {
            "n_sigma": self.n_sigma,
            "n_sigma_ul": self.n_sigma_ul,
            "sed_type_init": "likelihood",
        }

        table = table_from_row_data(rows=rows, meta=meta)
        model = datasets.models[self.source]
        return FluxPoints.from_table(
            table=table,
            reference_model=model.copy(),
            gti=datasets.gti,
            format="gadf-sed",
        )

    def estimate_flux_point(self, datasets, energy_min, energy_max):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        datasets : `Datasets`
            Datasets
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds to compute the flux point for.

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        datasets_sliced = datasets.slice_by_energy(
            energy_min=energy_min, energy_max=energy_max
        )
        if self.sum_over_energy_groups:
            datasets_sliced = Datasets(
                [_.to_image(name=_.name) for _ in datasets_sliced]
            )

        if len(datasets_sliced) > 0:
            datasets_sliced.models = datasets.models.copy()
            return super().run(datasets=datasets_sliced)
        else:
            log.warning(f"No dataset contribute in range {energy_min}-{energy_max}")
            model = datasets.models[self.source].spectral_model
            return self._nan_result(datasets, model, energy_min, energy_max)

    def _nan_result(self, datasets, model, energy_min, energy_max):
        """NaN result"""
        energy_axis = MapAxis.from_energy_edges([energy_min, energy_max])

        with np.errstate(invalid="ignore", divide="ignore"):
            result = model.reference_fluxes(energy_axis=energy_axis)
            # convert to scalar values
            result = {key: value.item() for key, value in result.items()}

        result.update(
            {
                "norm": np.nan,
                "stat": np.nan,
                "success": False,
                "norm_err": np.nan,
                "ts": np.nan,
                "counts": np.zeros(len(datasets)),
                "npred": np.nan * np.zeros(len(datasets)),
                "npred_excess": np.nan * np.zeros(len(datasets)),
                "datasets": datasets.names,
            }
        )

        if "errn-errp" in self.selection_optional:
            result.update({"norm_errp": np.nan, "norm_errn": np.nan})

        if "ul" in self.selection_optional:
            result.update({"norm_ul": np.nan})

        if "scan" in self.selection_optional:
            norm = super()._set_norm_parameter()
            norm_scan = norm.scan_values
            result.update({"norm_scan": norm_scan, "stat_scan": np.nan * norm_scan})

        return result

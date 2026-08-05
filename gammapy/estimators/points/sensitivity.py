# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import logging
from itertools import repeat
from astropy.table import Column, Table
import astropy.units as u
from gammapy.maps import Map, MapAxis
from gammapy.datasets import Datasets
from gammapy.datasets.actors import DatasetsActor
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.stats import WStatCountsStatistic
import gammapy.utils.parallel as parallel

from ..core import Estimator
from ..utils import apply_threshold_sensitivity
from .sed import FluxPointsEstimator

log = logging.getLogger(__name__)

__all__ = ["SensitivityEstimator", "JointSensitivityEstimator"]


class SensitivityEstimator(Estimator):
    """Estimate sensitivity.

    This class allows to determine for each reconstructed energy bin the flux
    associated to the number of gamma-ray events for which the significance is
    ``n_sigma``, and being larger than ``gamma_min`` and ``bkg_sys`` percent
    larger than the number of background events in the ON region.


    Parameters
    ----------
    spectral_model : `~gammapy.modeling.models.SpectralModel`, optional
        Spectral model assumption. Default is power-law with spectral index of 2.
    n_sigma : float, optional
        Minimum significance. Default is 5.
    gamma_min : float, optional
        Minimum number of gamma-rays. Default is 10.
    bkg_syst_fraction : float, optional
        Fraction of background counts above which the number of gamma-rays is. Default is 0.05.

    Examples
    --------
    For a usage example see :doc:`/tutorials/analysis-1d/cta_sensitivity` tutorial.

    """

    tag = "SensitivityEstimator"

    def __init__(
        self,
        spectral_model=None,
        n_sigma=5.0,
        gamma_min=10,
        bkg_syst_fraction=0.05,
    ):
        if spectral_model is None:
            spectral_model = PowerLawSpectralModel(
                index=2, amplitude="1 cm-2 s-1 TeV-1"
            )

        self.spectral_model = spectral_model
        self.n_sigma = n_sigma
        self.gamma_min = gamma_min
        self.bkg_syst_fraction = bkg_syst_fraction

    def estimate_min_excess(self, dataset):
        """Estimate minimum excess to reach the given significance.

        Parameters
        ----------
        dataset : `~gammapy.datasets.SpectrumDataset`
            Spectrum dataset.

        Returns
        -------
        excess : `~gammapy.maps.RegionNDMap`
            Minimal excess.
        """
        n_off = dataset.counts_off.data

        stat = WStatCountsStatistic(
            n_on=dataset.alpha.data * n_off, n_off=n_off, alpha=dataset.alpha.data
        )
        excess_counts = stat.n_sig_matching_significance(self.n_sigma)

        excess_counts = apply_threshold_sensitivity(
            dataset.background.data,
            excess_counts,
            self.gamma_min,
            self.bkg_syst_fraction,
        )

        excess = Map.from_geom(geom=dataset._geom, data=excess_counts)
        return excess

    def estimate_min_e2dnde(self, excess, dataset):
        """Estimate e2dnde from a given minimum excess.

        Parameters
        ----------
        excess : `~gammapy.maps.RegionNDMap`
            Minimal excess.
        dataset : `~gammapy.datasets.SpectrumDataset`
            Spectrum dataset.

        Returns
        -------
        e2dnde : `~astropy.units.Quantity`
            Minimal differential flux.
        """
        energy = dataset._geom.axes["energy"].center

        dataset.models = SkyModel(spectral_model=self.spectral_model)
        npred = dataset.npred_signal()

        phi_0 = excess / npred

        dnde_model = self.spectral_model(energy=energy)
        e2dnde = phi_0.data[:, 0, 0] * dnde_model * energy**2
        return e2dnde.to("erg / (cm2 s)")

    def _get_criterion(self, excess, bkg):
        is_gamma_limited = excess == self.gamma_min
        is_bkg_syst_limited = excess == bkg * self.bkg_syst_fraction
        criterion = np.empty(excess.shape, dtype="U12")
        criterion[is_gamma_limited] = "gamma"
        criterion[is_bkg_syst_limited] = "bkg"
        criterion[~np.logical_or(is_gamma_limited, is_bkg_syst_limited)] = (
            "significance"
        )
        return criterion

    def run(self, dataset):
        """Run the sensitivity estimation.

        Parameters
        ----------
        dataset : `~gammapy.datasets.SpectrumDatasetOnOff`
            Dataset to compute sensitivity for.

        Returns
        -------
        sensitivity : `~astropy.table.Table`
            Sensitivity table. Containing the following columns:

                * e_ref : energy center
                * e_min : minimum energy values
                * e_max : maximum energy values
                * e2dnde : minimal differential flux
                * excess : number of excess counts in the bin
                * background : number of background counts in the bin
                * criterion : sensitivity-limiting criterion

        """
        energy = dataset._geom.axes["energy"].center

        if np.any(self.spectral_model(energy).value < 0.0):
            log.warning(
                "Spectral model predicts negative flux. Results of estimator should be interpreted with caution"
            )

        excess = self.estimate_min_excess(dataset)
        e2dnde = self.estimate_min_e2dnde(excess, dataset)
        criterion = self._get_criterion(
            excess.data.squeeze(), dataset.background.data.squeeze()
        )

        return Table(
            [
                Column(
                    data=energy,
                    name="e_ref",
                    format="5g",
                    description="Energy center",
                ),
                Column(
                    data=dataset._geom.axes["energy"].edges_min,
                    name="e_min",
                    format="5g",
                    description="Energy edge low",
                ),
                Column(
                    data=dataset._geom.axes["energy"].edges_max,
                    name="e_max",
                    format="5g",
                    description="Energy edge high",
                ),
                Column(
                    data=e2dnde,
                    name="e2dnde",
                    format="5g",
                    description="Energy squared times differential flux",
                ),
                Column(
                    data=np.atleast_1d(excess.data.squeeze()),
                    name="excess",
                    format="5g",
                    description="Number of excess counts in the bin",
                ),
                Column(
                    data=np.atleast_1d(dataset.background.data.squeeze()),
                    name="background",
                    format="5g",
                    description="Number of background counts in the bin",
                ),
                Column(
                    data=np.atleast_1d(criterion),
                    name="criterion",
                    description="Sensitivity-limiting criterion",
                ),
            ]
        )


class JointSensitivityEstimator(FluxPointsEstimator):
    """A joint sensitivity estimator.

    This class follows the logic of `~gammapy.estimators.FluxPointsEstimator`
    to compute sensitivity using Asimov datasets.
    This supports multiple telescopes, or multiple event types from the same telescope.
    All relevant models must be set on the datasets.

    Parameters
    ----------
    source : str or int
        For which source in the model to compute the flux points.
    n_sigma_sensitivity : float, optional
        Detection significance threshold. Default is 2.
    reoptimize : bool, optional
        If True, the free parameters of the other models are fitted in each bin independently,
        together with the norm of the source of interest
        (but the other parameters of the source of interest are kept frozen).
        If False, only the norm of the source of interest is fitted,
        and all other parameters are frozen at their current values.
        Default is False.
    energy_edges : `~astropy.units.Quantity`
        Energy bin edges./
    n_jobs : int, optional
        Number of processes used in parallel for the computation. The number of jobs is limited to the number of
        physical CPUs. If None, defaults to `~gammapy.utils.parallel.N_JOBS_DEFAULT`.
        Default is None.
    parallel_backend : {"multiprocessing", "ray"}, optional
        Which backend to use for multiprocessing. If None, defaults to `~gammapy.utils.parallel.BACKEND_DEFAULT`.


    """

    tag = "JointSensitivityEstimator"

    def __init__(
        self,
        source=0,
        n_sigma_sensitivity=2.0,
        reoptimize=False,
        energy_edges=[1, 10] * u.TeV,
        n_jobs=None,
        parallel_backend=None,
    ):
        super().__init__(
            energy_edges=energy_edges,
            source=source,
            n_sigma_sensitivity=n_sigma_sensitivity,
            reoptimize=reoptimize,
            n_jobs=n_jobs,
            selection_optional=["sensitivity"],
            parallel_backend=parallel_backend,
            allow_multiple_telescopes=True,
        )

    def _estimate_sensitivity_one_bin(self, datasets, energy_min, energy_max):
        """Estimate sensitivity for a single energy bin."""
        datasets_slice = datasets.slice_by_energy(energy_min, energy_max)
        if len(datasets_slice) == 0:
            return None

        models = datasets.models.copy()
        model = self.get_scale_model(models)

        models[self.source].spectral_model = model
        datasets_slice.models = models

        with datasets_slice.parameters.restore_status():
            if not self.reoptimize:
                datasets_slice.parameters.freeze_all()
                model.norm.frozen = False

            norm_sensitivity = self.estimate_sensitivity(datasets_slice, model.norm)[
                "norm_sensitivity"
            ]

        energy_axis = MapAxis.from_energy_edges(energy_edges=[energy_min, energy_max])
        ref_fluxes = model.reference_fluxes(energy_axis=energy_axis)
        e2dnde_sensitivity = norm_sensitivity * ref_fluxes["ref_e2dnde"]
        dnde_sensitivity = norm_sensitivity * ref_fluxes["ref_dnde"]

        return {
            "e_ref": ref_fluxes["e_ref"],
            "e_min": energy_min,
            "e_max": energy_max,
            "e2dnde": e2dnde_sensitivity.to("erg / (cm2 s)"),
            "dnde": dnde_sensitivity.to("1 / (TeV cm2 s)"),
            "norm_sensitivity": norm_sensitivity,
        }

    def run(self, datasets):
        """Run sensitivity estimation over all energy bins.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets` or list
            Input datasets.

        Returns
        -------
        table : `~astropy.table.Table`
            Sensitivity table with columns: ``e_ref``, ``e_min``, ``e_max``,
            ``e2dnde``, ``dnde``, ``norm_sensitivity``.
        """
        if not isinstance(datasets, (Datasets, DatasetsActor)):
            datasets = Datasets(datasets)

        if not datasets.energy_axes_are_aligned:
            raise ValueError("All datasets must have aligned energy axes.")

        rows = parallel.run_multiprocessing(
            self._estimate_sensitivity_one_bin,
            zip(
                repeat(datasets),
                self.energy_edges[:-1],
                self.energy_edges[1:],
            ),
            backend=self.parallel_backend,
            pool_kwargs=dict(processes=self.n_jobs),
            task_name="Energy bins",
        )

        rows = [r for r in rows if r is not None]
        return Table(rows, meta={"n_sigma_sensitivity": self.n_sigma_sensitivity})

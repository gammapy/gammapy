"""Tools to create profiles (i.e. 1D "slices" from 2D images)."""
import numpy as np
from astropy import units as u
from regions import CircleAnnulusSkyRegion, RectangleSkyRegion
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.maps import MapAxis
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic
from gammapy.utils.table import table_from_row_data
from .core import Estimator

__all__ = ["ExcessProfileEstimator"]


class ExcessProfileEstimator(Estimator):
    """Estimate profile from a DataSet.

    Parameters
    ----------
    regions : list of `regions`
        regions to use
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the profiles to be computed.
    n_sigma : float (optional)
        Number of sigma to compute errors. By default, it is 1.
    n_sigma_ul : float (optional)
        Number of sigma to compute upper limit. By default, it is 3.
    spectrum : `~gammapy.modeling.models.SpectralModel` (optional)
        Spectral model to compute the fluxes or brightness.
        Default is power-law with spectral index of 2.
    selection_optional : list of str
        Additional quantities to be estimated. Possible options are:

            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.

        By default all quantities are estimated.

    Examples
    --------
    This example shows how to compute a counts profile for the Fermi galactic
    center region::

        import matplotlib.pyplot as plt
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from gammapy.data import GTI
        from gammapy.estimators import ExcessProfileEstimator, ImageProfile
        from gammapy.utils.regions import make_orthogonal_rectangle_sky_regions
        from gammapy.datasets import Datasets

        # load example data
        datasets = Datasets.read("$GAMMAPY_DATA/fermi-3fhl-crab/",
            "Fermi-LAT-3FHL_datasets.yaml", "Fermi-LAT-3FHL_models.yaml")
        # configuration
        datasets[0].gti = GTI.create("0s", "1e7s", "2010-01-01")

        # creation of the boxes and axis
        start_line = SkyCoord(182.5, -5.8, unit='deg', frame='galactic')
        end_line = SkyCoord(186.5, -5.8, unit='deg', frame='galactic')
        boxes, axis = make_orthogonal_rectangle_sky_regions(start_line,
                                        end_line,
                                        datasets[0].counts.geom.wcs,
                                        1.*u.deg,
                                        11)

        # set up profile estimator and run
        prof_maker = ExcessProfileEstimator(boxes, axis)
        fermi_prof = prof_maker.run(datasets[0])

        # smooth and plot the data using the ImageProfile class
        fermi_prof.peek()
        plt.show()

        ax = plt.gca()
        ax.set_yscale('log')
        ax = fermi_prof.plot("flux", ax=ax)

    """

    tag = "ExcessProfileEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        regions,
        energy_edges=None,
        spectrum=None,
        n_sigma=1.0,
        n_sigma_ul=3.0,
        selection_optional="all",
    ):
        self.regions = regions
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul

        self.energy_edges = (
            u.Quantity(energy_edges) if energy_edges is not None else None
        )

        if spectrum is None:
            spectrum = PowerLawSpectralModel()

        self.spectrum = spectrum
        self.selection_optional = selection_optional

    def get_spectrum_datasets(self, dataset):
        """ Utility to make the final `~gammapy.datasts.Datasets`

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            the dataset to use for profile extraction
        Returns
        --------
        sp_datasets : array of `~gammapy.datasets.SpectrumDataset`
            the list of `~gammapy.datasets.SpectrumDataset` computed in each box
        """
        datasets = Datasets()

        for reg in self.regions:
            spectrum_dataset = dataset.to_spectrum_dataset(reg)
            datasets.append(spectrum_dataset)

        return datasets

    def _get_projected_distance(self):
        distances = []
        center = self.regions[0].center

        for idx, region in enumerate(self.regions):
            if isinstance(region, CircleAnnulusSkyRegion):
                distance = (region.inner_radius + region.outer_radius) / 2.0
            else:
                distance = center.separation(region.center)

            distances.append(distance)

        return MapAxis.from_nodes(
            u.Quantity(distances, "deg"), name="projected distance"
        )

    def make_prof(self, sp_datasets):
        """ Utility to make the profile in each region

        Parameters
        ----------
        sp_datasets : `~gammapy.datasets.MapDatasets` of `~gammapy.datasets.SpectrumDataset` or \
        `~gammapy.datasets.SpectrumDatasetOnOff`
            the dataset to use for profile extraction

        Returns
        --------
        results : list of dictionary
            the list of results (list of keys: x_min, x_ref, x_max, alpha, counts, background, excess, ts, sqrt_ts, \
            err, errn, errp, ul, exposure, solid_angle)
        """
        results = []

        distance = self._get_projected_distance()

        for index, spds in enumerate(sp_datasets):
            old_model = None
            if spds.models is not None:
                old_model = spds.models
            spds.models = SkyModel(spectral_model=self.spectrum)
            e_reco = spds.counts.geom.axes["energy"].edges

            # ToDo: When the function to_spectrum_dataset will manage the masks, use the following line
            # mask = spds.mask if spds.mask is not None else slice(None)
            mask = slice(None)
            if isinstance(spds, SpectrumDatasetOnOff):
                stats = WStatCountsStatistic(
                    spds.counts.data[mask][:, 0, 0],
                    spds.counts_off.data[mask][:, 0, 0],
                    spds.alpha.data[mask][:, 0, 0],
                )

            else:
                stats = CashCountsStatistic(
                    spds.counts.data[mask][:, 0, 0],
                    spds.npred_background().data[mask][:, 0, 0],
                )

            result = {
                "x_min": distance.edges[index],
                "x_max": distance.edges[index + 1],
                "x_ref": distance.center[index],
                "energy_edge": e_reco,
            }
            if isinstance(spds, SpectrumDatasetOnOff):
                result["alpha"] = stats.alpha
            result.update(
                {
                    "counts": stats.n_on,
                    "background": stats.n_bkg,
                    "excess": stats.n_sig,
                }
            )

            result["ts"] = stats.ts
            result["sqrt_ts"] = stats.sqrt_ts

            result["err"] = stats.error * self.n_sigma

            if "errn-errp" in self.selection_optional:
                result["errn"] = stats.compute_errn(self.n_sigma)
                result["errp"] = stats.compute_errp(self.n_sigma)

            if "ul" in self.selection_optional:
                result["ul"] = stats.compute_upper_limit(self.n_sigma_ul)

            npred = spds.npred().data[mask][:, 0, 0]
            e_reco_lo = e_reco[:-1]
            e_reco_hi = e_reco[1:]
            flux = (
                stats.n_sig
                / npred
                * spds.models[0].spectral_model.integral(e_reco_lo, e_reco_hi).value
            )
            result["flux"] = flux

            result["flux_err"] = stats.error / stats.n_sig * flux

            if "errn-errp" in self.selection_optional:
                result["flux_errn"] = np.abs(result["errn"]) / stats.n_sig * flux
                result["flux_errp"] = result["errp"] / stats.n_sig * flux

            if "ul" in self.selection_optional:
                result["flux_ul"] = result["ul"] / stats.n_sig * flux

            solid_angle = spds.counts.geom.solid_angle()
            result["solid_angle"] = (
                np.full(result["counts"].shape, solid_angle.to_value("sr")) * u.sr
            )

            results.append(result)
            if old_model is not None:
                spds.models = old_model

        return results

    def run(self, dataset):
        """Make the profiles

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            the dataset to use for profile extraction

        Returns
        --------
        imageprofile : `~gammapy.estimators.ImageProfile`
            Return an image profile class containing the result
        """
        if self.energy_edges is not None:
            axis = MapAxis.from_energy_edges(self.energy_edges)
            dataset = dataset.resample_energy_axis(energy_axis=axis)
        else:
            dataset = dataset.to_image()

        spectrum_datasets = self.get_spectrum_datasets(dataset)

        results = self.make_prof(spectrum_datasets)
        table = table_from_row_data(results)
        if isinstance(self.regions[0], RectangleSkyRegion):
            table.meta["PROFILE_TYPE"] = "orthogonal_rectangle"
        table.meta["SPECTRAL_MODEL"] = self.spectrum.to_dict()

        # return ImageProfile(table)
        return table

"""Tools to create profiles (i.e. 1D "slices" from 2D images)."""
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion
from gammapy.utils.table import table_from_row_data
from gammapy.stats import WStatCountsStatistic, CashCountsStatistic
from gammapy.datasets import SpectrumDatasetOnOff, Datasets
from gammapy.maps import MapAxis
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel
from .core import Estimator

__all__ = ["ExcessProfileEstimator"]


class ExcessProfileEstimator(Estimator):
    """Estimate profile from a DataSet.

    Parameters
    ----------
    regions : list of `regions`
        regions to use
    n_sigma : float (optional)
        Number of sigma to compute errors. By default, it is 1.
    n_sigma_ul : float (optional)
        Number of sigma to compute upper limit. By default, it is 3.
    spectrum : `~gammapy.modeling.models.SpectralModel` (optional)
        Spectral model to compute the fluxes or brightness.
        Default is power-law with spectral index of 2.

    Example
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

    def __init__(self, regions, spectrum=None, n_sigma=1., n_sigma_ul=3.):
        self.regions = regions
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul

        if spectrum is None:
            spectrum = PowerLawSpectralModel()

        self.spectrum = spectrum

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
        centers = []

        for region in self.regions:
            centers.append(region.center)

        centers = SkyCoord(centers)

        distance = centers.separation(centers[0])

        return MapAxis.from_nodes(distance, name="projected distance")

    def make_prof(self, sp_datasets, steps):
        """ Utility to make the profile in each region

        Parameters
        ----------
        sp_datasets : `~gammapy.datasets.MapDatasets` of `~gammapy.datasets.SpectrumDataset` or \
        `~gammapy.datasets.SpectrumDatasetOnOff`
            the dataset to use for profile extraction
        steps : list of str
            the steps to be used.
        Returns
        --------
        results : list of dictionary
            the list of results (list of keys: x_min, x_ref, x_max, alpha, counts, background, excess, ts, sqrt_ts, \
            err, errn, errp, ul, exposure, solid_angle)
        """
        if steps == "all":
            steps = ["errn-errp", "ul"]

        results = []

        distance = self._get_projected_distance()

        for index, spds in enumerate(sp_datasets):
            old_model = None
            if spds.models is not None:
                old_model = spds.models
            spds.models = SkyModel(spectral_model=self.spectrum)
            e_reco = spds.counts.geom.get_axis_by_name("energy").edges

            # ToDo: When the function to_spectrum_dataset will manage the masks, use the following line
            # mask = spds.mask if spds.mask is not None else slice(None)
            mask = slice(None)
            if isinstance(spds, SpectrumDatasetOnOff):
                stats = WStatCountsStatistic(
                    spds.counts.data[mask][:, 0, 0],
                    spds.counts_off.data[mask][:, 0, 0],
                    spds.alpha.data[mask][:, 0, 0]
                )

            else:
                stats = CashCountsStatistic(
                    spds.counts.data[mask][:, 0, 0],
                    spds.background.data[mask][:, 0, 0],
                )

            result = {
                "x_min": distance.edges[index],
                "x_max": distance.edges[index + 1],
                "x_ref": distance.center[index],
                "energy_edge": e_reco
            }
            if isinstance(spds, SpectrumDatasetOnOff):
                result["alpha"] = stats.alpha
            result.update({
                "counts": stats.n_on,
                "background": stats.background,
                "excess": stats.excess,
            })

            result["ts"] = stats.delta_ts
            result["sqrt_ts"] = stats.significance

            result["err"] = stats.error * self.n_sigma

            if "errn-errp" in steps:
                result["errn"] = stats.compute_errn(self.n_sigma)
                result["errp"] = stats.compute_errp(self.n_sigma)

            if "ul" in steps:
                result["ul"] = stats.compute_upper_limit(self.n_sigma_ul)

            npred = spds.npred_sig().data[mask][:, 0, 0]
            e_reco_lo = e_reco[:-1]
            e_reco_hi = e_reco[1:]
            flux = stats.excess / npred * \
                            spds.models[0].spectral_model.integral(e_reco_lo, e_reco_hi).value
            result["flux"] = flux

            result["flux_err"] = stats.error / stats.excess * flux

            if "errn-errp" in steps:
                result["flux_errn"] = np.abs(result["errn"]) / stats.excess * flux
                result["flux_errp"] = result["errp"] / stats.excess * flux

            if "ul" in steps:
                result["flux_ul"] = result["ul"] / stats.excess * flux

            solid_angle = spds.counts.geom.solid_angle()
            result["solid_angle"] = np.full(result['counts'].shape, solid_angle.to_value("sr")) * u.sr

            results.append(result)
            if old_model is not None:
                spds.models = old_model

        return results

    def run(self, dataset, steps="all"):
        """Make the profiles

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            the dataset to use for profile extraction
        steps : list of str
            Additional quantities to be estimated. Possible options are:

                * "errn-errp": estimate asymmetric errors.
                * "ul": estimate upper limits.

            By default all quantities are estimated.
            
        Returns
        --------
        imageprofile : `~gammapy.estimators.ImageProfile`
            Return an image profile class containing the result
        """
        spectrum_datasets = self.get_spectrum_datasets(dataset)
        results = self.make_prof(spectrum_datasets, steps)
        table = table_from_row_data(results)
        if isinstance(self.regions[0], RectangleSkyRegion):
            table.meta["PROFILE_TYPE"] = "orthogonal_rectangle"
        table.meta["SPECTRAL_MODEL"] = self.spectrum.to_dict()

        # return ImageProfile(table)
        return table

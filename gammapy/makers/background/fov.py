# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FoV background estimation."""
import logging
from gammapy.datasets import Datasets
from gammapy.modeling import Fit
from gammapy.modeling.models import FoVBackgroundModel, Model
from ..core import Maker

__all__ = ["FoVBackgroundMaker"]

log = logging.getLogger(__name__)


class FoVBackgroundMaker(Maker):
    """Normalize template background on the whole field-of-view.

    The dataset background model can be simply scaled (method="scale") or fitted (method="fit")
    on the dataset counts.

    The normalization is performed outside the exclusion mask that is passed on init.

    If a SkyModel is set on the input dataset and method is 'fit', its are frozen during
    the fov normalization fit.

    Parameters
    ----------
    method : str in ['fit', 'scale']
        the normalization method to be applied. Default 'scale'.
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        Exclusion mask
    spectral_model_tag : str
        Default norm spectral model to use for the `FoVBackgroundModel`, if none is defined
        on the dataset.
    """
    tag = "FoVBackgroundMaker"
    available_methods = ["fit", "scale"]

    def __init__(
        self, method="scale", exclusion_mask=None, spectral_model_tag="pl-norm"
    ):
        self.method = method
        self.exclusion_mask = exclusion_mask

        if "norm" not in spectral_model_tag:
            raise ValueError("Spectral model must be a norm spectral model")

        self.spectral_model_tag = spectral_model_tag

    @property
    def method(self):
        """Method"""
        return self._method

    @method.setter
    def method(self, value):
        """Method setter"""
        if value not in self.available_methods:
            raise ValueError(f"Not a valid method for FoVBackgroundMaker: {value}."
                             f" Choose from {self.available_methods}")

        self._method = value

    def make_default_fov_background_model(self, dataset):
        """Add fov background model to the model definition

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input map dataset.

        Returns
        -------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset including

        """
        spectral_model = Model.create(
            tag=self.spectral_model_tag, model_type="spectral"
        )

        bkg_model = FoVBackgroundModel(
            dataset_name=dataset.name, spectral_model=spectral_model.copy()
        )

        if dataset.models is None:
            dataset.models = bkg_model
        else:
            dataset.models = dataset.models + bkg_model

        return dataset

    def run(self, dataset, observation=None):
        """Run FoV background maker.

        Fit the background model norm

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input map dataset.

        """
        mask_fit = dataset.mask_fit

        if self.exclusion_mask:
            geom = dataset.counts.geom
            dataset.mask_fit = self.exclusion_mask.interp_to_geom(geom=geom)

        if dataset.background_model is None:
            dataset = self.make_default_fov_background_model(dataset)

        if self.method == "fit":
            dataset = self.make_background_fit(dataset)
        else:
            # always scale the background first
            dataset = self.make_background_scale(dataset)

        dataset.mask_fit = mask_fit
        return dataset

    @staticmethod
    def make_background_fit(dataset):
        """Fit the FoV background model on the dataset counts data

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.

        Returns
        -------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset with fitted background model
        """
        # freeze all model components not related to background model

        models = dataset.models

        with models.restore_status(restore_values=False):
            models.select(tag="sky-model").freeze()

            fit = Fit([dataset])
            fit_result = fit.run()
            if not fit_result.success:
                log.info(f"Fit did not converge for {dataset.name}.")

        return dataset

    @staticmethod
    def make_background_scale(dataset):
        """Fit the FoV background model on the dataset counts data

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset.

        Returns
        -------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset with scaled background model

        """
        mask = dataset.mask
        count_tot = dataset.counts.data[mask].sum()
        bkg_tot = dataset.npred_background().data[mask].sum()

        if count_tot <= 0.0:
            log.warning(
                f"FoVBackgroundMaker failed. No counts found outside exclusion mask for {dataset.name}."
            )
        elif bkg_tot <= 0.0:
            log.warning(
                f"FoVBackgroundMaker failed. No positive background found outside exclusion mask for {dataset.name}."
            )
        else:
            value = count_tot / bkg_tot
            dataset.models[f"{dataset.name}-bkg"].spectral_model.norm.value = value

        return dataset

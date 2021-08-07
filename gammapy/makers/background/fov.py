# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FoV background estimation."""
import logging
import numpy as np
from gammapy.maps import Map
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

    If a SkyModel is set on the input dataset and method is 'fit', it' parameters
    are frozen during the fov normalization fit.

    If the requirement (greater than) of either min_counts or min_npred_background is not satisfied,
    the background will not be normalised

    Parameters
    ----------
    method : str in ['fit', 'scale']
        the normalization method to be applied. Default 'scale'.
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        Exclusion mask
    spectral_model : SpectralModel or str
        Reference norm spectral model to use for the `FoVBackgroundModel`, if none is defined
        on the dataset. By default, use pl-norm.
    min_counts : int
        Minimum number of counts required outside the exclusion region
    min_npred_background : float
       Minimum number of predicted background counts required outside the exclusion region
    """

    tag = "FoVBackgroundMaker"
    available_methods = ["fit", "scale"]

    def __init__(
        self,
        method="scale",
        exclusion_mask=None,
        spectral_model="pl-norm",
        min_counts=0,
        min_npred_background=0,
        fit=None
    ):
        self.method = method
        self.exclusion_mask = exclusion_mask
        self.min_counts = min_counts
        self.min_npred_background = min_npred_background

        if isinstance(spectral_model, str):
            spectral_model = Model.create(tag=spectral_model, model_type="spectral")

        if not spectral_model.is_norm_spectral_model:
            raise ValueError("Spectral model must be a norm spectral model")

        self.default_spectral_model = spectral_model

        if fit is None:
            fit = Fit()

        self.fit = fit

    @property
    def method(self):
        """Method"""
        return self._method

    @method.setter
    def method(self, value):
        """Method setter"""
        if value not in self.available_methods:
            raise ValueError(
                f"Not a valid method for FoVBackgroundMaker: {value}."
                f" Choose from {self.available_methods}"
            )

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
            Map dataset including background model

        """
        bkg_model = FoVBackgroundModel(
            dataset_name=dataset.name, spectral_model=self.default_spectral_model.copy()
        )

        if dataset.models is None:
            dataset.models = bkg_model
        else:
            dataset.models = dataset.models + bkg_model

        return dataset

    def make_exclusion_mask(self, dataset):
        """Project input exclusion mask to dataset geom

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input map dataset.

        Returns
        -------
        mask : `~gammapy.maps.WcsNDMap`
            Projected exclusion mask
        """
        geom = dataset._geom
        if self.exclusion_mask:
            mask = self.exclusion_mask.interp_to_geom(geom=geom)
        else:
            mask = Map.from_geom(geom=geom, data=1, dtype=bool)
        return mask

    def run(self, dataset, observation=None):
        """Run FoV background maker.

        Fit the background model norm

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input map dataset.

        """
        mask_fit = dataset.mask_fit

        dataset.mask_fit = self.make_exclusion_mask(dataset)

        if dataset.background_model is None:
            dataset = self.make_default_fov_background_model(dataset)

        if self.method == "fit":
            dataset = self.make_background_fit(dataset)
        else:
            # always scale the background first
            dataset = self.make_background_scale(dataset)

        dataset.mask_fit = mask_fit
        return dataset

    def make_background_fit(self, dataset):
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

        models = dataset.models.select(tag="sky-model")

        with models.restore_status(restore_values=False):
            models.select(tag="sky-model").freeze()

            fit_result = self.fit.run(datasets=[dataset])
            if not fit_result["optimize_result"].success:
                log.warning(
                    f"FoVBackgroundMaker failed. Fit did not converge for {dataset.name}. "
                    f"Setting mask to False."
                )
                dataset.mask_safe.data[...] = False

        return dataset

    def make_background_scale(self, dataset):
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

        if count_tot <= self.min_counts:
            log.warning(
                f"FoVBackgroundMaker failed. Only {int(count_tot)} counts outside exclusion mask for {dataset.name}. "
                f"Setting mask to False."
            )
            dataset.mask_safe.data[...] = False
        elif bkg_tot <= self.min_npred_background:
            log.warning(
                f"FoVBackgroundMaker failed. Only {int(bkg_tot)} background counts outside exclusion mask for {dataset.name}. "
                f"Setting mask to False."
            )
            dataset.mask_safe.data[...] = False
        else:
            value = count_tot / bkg_tot
            err = np.sqrt(count_tot) / bkg_tot
            dataset.models[f"{dataset.name}-bkg"].spectral_model.norm.value = value
            dataset.models[f"{dataset.name}-bkg"].spectral_model.norm.error = err

        return dataset

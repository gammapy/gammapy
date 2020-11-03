# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FoV background estimation."""
import logging
from gammapy.datasets import Datasets
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

    def __init__(
        self, method="scale", exclusion_mask=None, spectral_model_tag="pl-norm"
    ):
        if method in ["fit", "scale"]:
            self.method = method
        else:
            raise ValueError(f"Not a valid method for FoVBackgroundMaker: {method}.")

        self.exclusion_mask = exclusion_mask

        if "norm" not in spectral_model_tag:
            raise ValueError("Spectral model must be a norm spectral model")

        self.spectral_model_tag = spectral_model_tag

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
        dataset.mask_fit = self._reproject_exclusion_mask(dataset)

        if dataset.background_model is None:
            dataset = self.make_default_fov_background_model(dataset)

        if self.method == "fit":
            self._fit_bkg(dataset)
        else:
            # always scale the background first
            self._scale_bkg(dataset)

        dataset.mask_fit = mask_fit
        return dataset

    def _reproject_exclusion_mask(self, dataset):
        """Reproject the exclusion on the dataset geometry"""
        mask_map = Map.from_geom(dataset.counts.geom)
        if self.exclusion_mask is not None:
            coords = dataset.counts.geom.get_coord()
            vals = self.exclusion_mask.get_by_coord(coords)
            mask_map.data += vals
        else:
            mask_map.data[...] = 1

        return mask_map.data.astype("bool")

    def _fit_bkg(self, dataset):
        """Fit the FoV background model on the dataset counts data"""
        # freeze all model components not related to background model
        datasets = Datasets([dataset])

        parameters_frozen = []
        for par in datasets.parameters:
            parameters_frozen.append(par.frozen)
            if par not in dataset.background_model.parameters:
                par.frozen = True

        fit = Fit(datasets)
        fit_result = fit.run()
        if not fit_result.success:
            log.info(f"Fit did not converge for {dataset.name}.")

        # Unfreeze parameters
        for idx, par in enumerate(datasets.parameters):
            par.frozen = parameters_frozen[idx]

    def _scale_bkg(self, dataset):
        """Fit the FoV background model on the dataset counts data"""
        mask = dataset.mask
        count_tot = dataset.counts.data[mask].sum()
        bkg_tot = dataset.npred_background().data[mask].sum()

        if count_tot <= 0.0:
            log.info(
                f"FoVBackgroundMaker failed. No counts found outside exclusion mask for {dataset.name}."
            )
        elif bkg_tot <= 0.0:
            log.info(
                f"FoVBackgroundMaker failed. No positive background found outside exclusion mask for {dataset.name}."
            )
        else:
            value = count_tot / bkg_tot
            dataset.models[f"{dataset.name}-bkg"].spectral_model.norm.value = value

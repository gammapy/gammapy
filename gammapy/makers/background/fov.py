# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FoV background estimation."""
import logging
from gammapy.maps import Map
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
    """
    tag = "FoVBackgroundMaker"

    def __init__(self, method="scale", exclusion_mask=None):
        if method in ["fit", "scale"]:
            self.method = method
        else:
            raise ValueError(f"Incorrect method for FoVBackgroundMaker: {method}.")
        self.exclusion_mask = exclusion_mask

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

        if self.method is "fit":
            self._fit_bkg(dataset)
        else:
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
        from gammapy.modeling import Fit
        from gammapy.datasets import Datasets

        # freeze all model components not related to background model
        datasets = Datasets([dataset])

        parameters_frozen = []
        for par in datasets.parameters:
            parameters_frozen.append(par.frozen)
            if par not in dataset.background_model.parameters:
                par.frozen = True

        fit = Fit(datasets)
        fit_result = fit.run()
        if fit_result.success is False:
            log.info(
                f"FoVBackgroundMaker failed. No fit convergence for {dataset.name}."
            )

        # Unfreeze parameters
        for idx, par in enumerate(datasets.parameters):
            par.frozen = parameters_frozen[idx]

    def _scale_bkg(self, dataset):
        """Fit the FoV background model on the dataset counts data"""
        mask = dataset.mask
        count_tot = dataset.counts.data[mask].sum()
        bkg_tot = dataset.background_model.map.data[mask].sum()

        if count_tot <= 0.0:
            log.info(
                f"FoVBackgroundMaker failed. No counts found outside exclusion mask for {dataset.name}."
            )
        elif bkg_tot <= 0.0:
            log.info(
                f"FoVBackgroundMaker failed. No positive background found outside exclusion mask for {dataset.name}."
            )
        else:
            scale = count_tot / bkg_tot
            dataset.background_model.norm.value = scale

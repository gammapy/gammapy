# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FoV background estimation."""
import logging
from gammapy.maps import Map
from gammapy.modeling import Fit

__all__ = ["FoVBackgroundMaker"]

log = logging.getLogger(__name__)

class FoVBackgroundMaker:
    """Normalize template background on the whole field-of-view.

    Parameters
    ----------
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        Exclusion mask
    """
    def __init__(self, exclusion_mask=None):
        self.exclusion_mask = exclusion_mask

    def run(self, dataset):
        """Run FoV background maker.

        Fit the background model norm

        Parameters
        ----------
        dataset : `~gammapy.cube.fit.MapDataset`
            Input map dataset.

        """
        mask_fit = dataset.mask_fit

        mask_map = Map.from_geom(dataset.counts.geom)
        if self.exclusion_mask is not None:
            coords = dataset.counts.geom.get_coord()
            vals = self.exclusion_mask.get_by_coord(coords)
            mask_map.data += vals

        dataset.mask_fit = mask_map.data.astype('bool')

        # Here we assume that the model is only the background model
        # TODO : freeze all model components not related to background model?
        fit = Fit([dataset])
        fit_result = fit.run()
        if fit_result.success == False:
            log.info("FoVBackgroundMaker failed. No fit convergence for {dataset.name}.")

        dataset.mask_fit = mask_fit
        return dataset
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FoV background estimation."""
import numpy as np
from gammapy.modeling import Fit

__all__ = ["FoVBackgroundMaker"]


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
        mask_safe = dataset.mask_safe

        # Here we assume that the model is only the background model
        # TODO : freeze all model components not related to background model?
        fit = Fit([dataset])
        fit_result = fit.run()
        if fit_result.success == False:
            print("FoVBackgroundMaker failed. No fit convergence.")

        return dataset
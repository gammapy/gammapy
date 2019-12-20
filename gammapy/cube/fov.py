# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FoV background estimation."""
import numpy as np
from astropy.convolution import Ring2DKernel, Tophat2DKernel
from astropy.coordinates import Angle
from gammapy.cube.fit import MapDatasetOnOff
from gammapy.maps import Map, scale_cube

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
         
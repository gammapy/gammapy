# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy.convolution import Tophat2DKernel
from gammapy.stats import significance, significance_on_off
from gammapy.datasets import MapDataset, MapDatasetOnOff

__all__ = [
    "LiMaMapEstimator",
]

log = logging.getLogger(__name__)


class LiMaMapEstimator:
    """Computes correlated excess, significance for MapDatasets


    Parameters
    ----------
    correlation_radius : ~astropy.coordinate.Angle
        correlation radius to use
    """

    def __init__(self, correlation_radius):
        self.radius = Angle(correlation_radius)

    def run(self, dataset):
        """Compute correlated excess, Li & Ma significance and flux maps

        This requires datasets with only 1 energy bins (image-like).
        Usually you can obtain one with `dataset.to_image()`


        Parameters
        ----------
        dataset : `~gammapy.cube.MapDataset` or `~gammapy.cube.MapDataset`
            input dataset

        Returns
        -------
        images : dict
            Dictionary containing result maps. Keys are: significance,
            counts, background and excess for a MapDataset significance,
            n_on, background, excess, alpha otherwise

        """
        if not isinstance(dataset, MapDataset):
            raise ValueError("Unsupported dataset type")

        pixel_size = np.mean(np.abs(dataset.counts.geom.wcs.wcs.cdelt))
        size = self.radius.deg / pixel_size
        kernel = Tophat2DKernel(size)

        if isinstance(dataset, MapDatasetOnOff):
            result = self.__class__.compute_lima_on_off_image(
                dataset.counts,
                dataset.counts_off,
                dataset.acceptance,
                dataset.acceptance_off,
                kernel,
            )
        else:
            background = dataset.npred()
            result = self.__class__.compute_lima_image(dataset.counts, background, kernel)
        return result

    @staticmethod
    def compute_lima_image(counts, background, kernel):
        """Compute Li & Ma significance and flux images for known background.

        Parameters
        ----------
        counts : `~gammapy.maps.WcsNDMap`
            Counts image
        background : `~gammapy.maps.WcsNDMap`
            Background image
        kernel : `astropy.convolution.Kernel2D`
            Convolution kernel

        Returns
        -------
        images : dict
            Dictionary containing result maps
            Keys are: significance, counts, background and excess

        See Also
        --------
        gammapy.stats.significance
        """
        # Kernel is modified later make a copy here
        kernel = copy.deepcopy(kernel)
        kernel.normalize("peak")

        # fft convolution adds numerical noise, to ensure integer results we call
        # np.rint
        counts_conv = np.rint(counts.convolve(kernel.array).data)
        background_conv = background.convolve(kernel.array).data
        excess_conv = counts_conv - background_conv
        significance_conv = significance(counts_conv, background_conv, method="lima")
        return {
            "significance": counts.copy(data=significance_conv),
            "counts": counts.copy(data=counts_conv),
            "background": counts.copy(data=background_conv),
            "excess": counts.copy(data=excess_conv),
        }


    @staticmethod
    def compute_lima_on_off_image(n_on, n_off, a_on, a_off, kernel):
        """Compute Li & Ma significance and flux images for on-off observations.

        Parameters
        ----------
        n_on : `~gammapy.maps.WcsNDMap`
            Counts image
        n_off : `~gammapy.maps.WcsNDMap`
            Off counts image
        a_on : `~gammapy.maps.WcsNDMap`
            Relative background efficiency in the on region
        a_off : `~gammapy.maps.WcsNDMap`
            Relative background efficiency in the off region
        kernel : `astropy.convolution.Kernel2D`
            Convolution kernel

        Returns
        -------
        images : dict
            Dictionary containing result maps
            Keys are: significance, n_on, background, excess, alpha

        See Also
        --------
        gammapy.stats.significance_on_off
        """
        # Kernel is modified later make a copy here
        kernel = copy.deepcopy(kernel)
        kernel.normalize("peak")

        # fft convolution adds numerical noise, to ensure integer results we call
        # np.rint
        n_on_conv = np.rint(n_on.convolve(kernel.array).data)

        with np.errstate(invalid="ignore", divide="ignore"):
            background = a_on / a_off
        background *= n_off
        background.data[a_off.data == 0] = 0.0
        background_conv = background.convolve(kernel.array).data

        n_off_conv = n_off.convolve(kernel.array).data

        with np.errstate(invalid="ignore", divide="ignore"):
            alpha_conv = background_conv / n_off_conv

        significance_conv = significance_on_off(
            n_on_conv, n_off_conv, alpha_conv, method="lima"
        )
        excess_conv = n_on_conv - background_conv

        return {
            "significance": n_on.copy(data=significance_conv),
            "n_on": n_on.copy(data=n_on_conv),
            "background": n_on.copy(data=background_conv),
            "excess": n_on.copy(data=excess_conv),
            "alpha": n_on.copy(data=alpha_conv),
        }

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy.convolution import Tophat2DKernel
from gammapy.stats import significance, significance_on_off
from gammapy.cube import MapDataset, MapDatasetOnOff

__all__ = ["SignificanceMapEstimator", "compute_lima_image", "compute_lima_on_off_image"]

log = logging.getLogger(__name__)


class SignificanceMapEstimator:
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
        """
        if isinstance(dataset, MapDataset):
            self._run_mapdataset(dataset)
        elif isinstance(dataset, MapDatasetOnOff):
            self._run_mapdataset_onoff(dataset)
        else:
            raise ValueError("Unsupported dataset type")

    def _run_mapdataset(self, dataset):
        """Apply Li & Ma with known background"""
        size = self.radius.deg/np.mean(dataset.counts.geom.wcs.wcs.cdelt)
        kernel = Tophat2DKernel(size)

        counts = dataset.counts
        background = dataset.npred()

        return compute_lima_image(counts, background, kernel)
    

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
    a_on_conv = a_on.convolve(kernel.array).data

    with np.errstate(invalid="ignore", divide="ignore"):
        alpha_conv = a_on_conv / a_off.data

    significance_conv = significance_on_off(
        n_on_conv, n_off.data, alpha_conv, method="lima"
    )

    with np.errstate(invalid="ignore"):
        background_conv = alpha_conv * n_off.data
    excess_conv = n_on_conv - background_conv

    return {
        "significance": n_on.copy(data=significance_conv),
        "n_on": n_on.copy(data=n_on_conv),
        "background": n_on.copy(data=background_conv),
        "excess": n_on.copy(data=excess_conv),
        "alpha": n_on.copy(data=alpha_conv),
    }

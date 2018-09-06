# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from copy import deepcopy
import logging
import numpy as np
from ..stats import significance, significance_on_off

__all__ = ["compute_lima_image", "compute_lima_on_off_image"]

log = logging.getLogger(__name__)


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
    kernel = deepcopy(kernel)
    kernel.normalize("peak")

    counts_conv = counts.convolve(kernel.array).data
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

    See also
    --------
    gammapy.stats.significance_on_off
    """
    # Kernel is modified later make a copy here
    kernel = deepcopy(kernel)
    kernel.normalize("peak")

    n_on_conv = n_on.convolve(kernel.array).data
    a_on_conv = a_on.convolve(kernel.array).data
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

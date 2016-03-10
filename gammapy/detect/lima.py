# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from copy import deepcopy
import logging

import numpy as np
from astropy.convolution import Tophat2DKernel

from ..image import SkyMapCollection
from ..stats import significance, significance_on_off

__all__ = ['compute_lima_map', 'compute_lima_on_off_map']

log = logging.getLogger(__name__)



def compute_lima_map(counts, background, kernel, exposure=None):
    """
    Compute Li&Ma significance and flux maps for known background.

    If exposure is given the corresponding flux map is computed and returned.  

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map
    background : `~numpy.ndarray`
        Background map
    kernel : `astropy.convolution.Kernel2D`
        convolution kernel. 
    exposure : `~numpy.ndarray`
        Exposure map

    Returns
    -------
    SkyMapCollection : `gammapy.data.maps.SkyMapCollection`
        Bunch of result maps.


    See Also
    --------
    gammapy.stats.significance
    """
    from scipy.ndimage import convolve

    # Kernel is modified later make a copy here
    kernel = deepcopy(kernel)

    if not kernel.is_bool:
        log.warn('Using weighted kernels can lead to biased results.')

    kernel.normalize('peak')
    counts_ = convolve(counts, kernel.array, mode='constant', cval=np.nan)
    background_ = convolve(background, kernel.array, mode='constant', cval=np.nan)

    significance_lima = significance(counts_, background_, method='lima') 

    result = SkyMapCollection(significance=significance_lima,
                       counts=counts_,
                       background=background_,
                       excess= counts_ - background_)

    if not exposure is None:
        kernel.normalize('integral')
        exposure_ = convolve(exposure, kernel.array, mode='constant', cval=np.nan)
        flux = (counts_ - background_) / exposure_
        result.flux = flux

    return result


def compute_lima_on_off_map(n_on, n_off, a_on, a_off, kernel, exposure=None):
    """
    Compute Li&Ma significance and flux maps for on-off observations.

    Parameters
    ----------
    n_on : `~numpy.ndarray`
        Counts map.
    n_off : `~numpy.ndarray`
        Off counts map.
    a_on : `~numpy.ndarray`
        Relative background efficiency in the on region
    a_off : `~numpy.ndarray`
        Relative background efficiency in the off region
    kernel : `astropy.convolution.Kernel2D`
        convolution kernel. 
    exposure : `~numpy.ndarray`
        Exposure map.
    
    Returns
    -------
    SkyMapCollection : `gammapy.data.maps.SkyMapCollection`
        Bunch of result maps.   

    See also
    --------
    gammapy.stats.significance_on_off

    """
    from scipy.ndimage import convolve

    # Kernel is modified later make a copy here
    kernel = deepcopy(kernel)

    if not kernel.is_bool:
        log.warn('Using weighted kernels can lead to biased results.')

    kernel.normalize('peak')
    n_on_ = convolve(n_on, kernel.array, mode='constant', cval=np.nan)
    a_ = convolve(a_on, kernel.array, mode='constant', cval=np.nan)
    alpha = a_ / a_off
    background = alpha * n_off

    significance_lima = significance_on_off(n_on_, n_off, alpha, method='lima')
    
    result = SkyMapCollection(significance=significance_lima,
                       n_on=n_on_,
                       background=background,
                       excess=n_on_ - background,
                       alpha=alpha)

    if not exposure is None:
        kernel.normalize('integral')
        exposure_ = convolve(exposure, kernel.array, mode='constant', cval=np.nan)
        flux = (n_on_ - background_) / exposure_
        result.flux = flux

    return result

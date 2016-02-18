# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging

import numpy as np
from astropy.convolution import Tophat2DKernel

from ..extern.bunch import Bunch
from ..stats import significance, significance_on_off

__all__ = ['compute_lima_map', 'compute_lima_on_off_map']

log = logging.getLogger(__name__)


def _fftconvolve_boundary_nan(array, kernel):
    """
    Wrapper for `~scipy.signal.fftconvolve` that sets all values, that would
    require any kind of boundary handling to NaN.
    """
    from scipy.signal import fftconvolve
    result = fftconvolve(array, kernel, mode='valid')
    padding = [ _ // 2 for _ in kernel.shape]
    return np.pad(result, (padding, padding), mode=str('constant'),
                  constant_values=np.nan)


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
    Bunch : `gammapy.extern.bunch.Bunch`
        Bunch of result maps.
    """
    
    if not kernel.is_bool:
        log.warn('Using weighted kernels can lead to biased results.')

    kernel.normalize('peak')
    counts_ = _fftconvolve_boundary_nan(counts, kernel.array)
    background_ = _fftconvolve_boundary_nan(background, kernel.array)

    significance_lima = significance(counts_, background_, method='lima') 

    if not exposure is None:
        kernel.normalize('integral')
        exposure_ = _fftconvolve_boundary_nan(exposure, kernel.array)
        flux = (counts_ - background_) / exposure_
        # TODO: return correlated counts and background map?   
        return Bunch(significance=significance_lima, flux=flux)

    return Bunch(significance=significance_lima)


def compute_lima_on_off_map(counts, background, kernel, alpha, exposure=None):
    """
    Compute Li&Ma significance and flux maps for known background.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map
    background : `~numpy.ndarray`
        Background map
    alpha : `~numpy.ndarray`
        Exposure ratio map.
    kernel : `astropy.convolution.Kernel2D`
        convolution kernel. 
    exposure : `~numpy.ndarray`
        Exposure map
    
    Returns
    -------
    Bunch : `gammapy.extern.bunch.Bunch`
        Bunch of result maps.   
    """
    

    flux = 0
    significance_lima = significance_on_off() 


    return Bunch(significance=significance, flux=flux, alpha=alpha)





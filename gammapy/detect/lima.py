# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from copy import deepcopy
import logging

import numpy as np
from astropy.convolution import Tophat2DKernel

from ..extern.bunch import Bunch
from ..stats import significance, significance_on_off

__all__ = ['compute_lima_map', 'compute_lima_on_off_map']

log = logging.getLogger(__name__)


def _convolve_boundary_nan(array, kernel, fft=True):
    """
    Wrapper for `~scipy.signal.convolve` an `~scipy.signal.fftconvolve` that
    sets all values in the results array, that would require any kind of
    boundary handling to NaN and thus leaves the total size of the resulting
    array unchanged.
    """
    from scipy.signal import convolve, fftconvolve

    _convolve = fftconvolve if fft else convolve

    result = _convolve(array, kernel, mode='valid')
    padding = [ _ // 2 for _ in kernel.shape]
    return np.pad(result, (padding, padding), mode=str('constant'),
                  constant_values=np.nan)


def compute_lima_map(counts, background, kernel, exposure=None, fft=False):
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
    fft : bool
        Use fast fft convolution. Default is False.
    
    Returns
    -------
    Bunch : `gammapy.extern.bunch.Bunch`
        Bunch of result maps.
    """
    # Kernel is modified later make a copy here
    kernel = deepcopy(kernel)

    if not kernel.is_bool:
        log.warn('Using weighted kernels can lead to biased results.')

    kernel.normalize('peak')
    counts_ = _convolve_boundary_nan(counts, kernel.array, fft=fft)
    background_ = _convolve_boundary_nan(background, kernel.array, fft=fft)

    significance_lima = significance(counts_, background_, method='lima') 

    result = Bunch(significance=significance_lima,
                   counts=counts_,
                   background=background_,
                   excess= counts_ - background_)

    if not exposure is None:
        kernel.normalize('integral')
        exposure_ = _convolve_boundary_nan(exposure, kernel.array, fft=fft)
        flux = (counts_ - background_) / exposure_
        result.flux = flux

    return result


def compute_lima_on_off_map(counts, off, a_on, a_off, kernel, exposure=None,
                            fft=False):
    """
    Compute Li&Ma significance and flux maps on off observation.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map
    off : `~numpy.ndarray`
        Off counts map.
    a_on : `~numpy.ndarray`
        Relative .
    a_off : `~numpy.ndarray`
        Exposure ratio map.
    kernel : `astropy.convolution.Kernel2D`
        convolution kernel. 
    exposure : `~numpy.ndarray`
        Exposure map.
    fft : bool
        Use fast fft convolution. Default is False.
    
    Returns
    -------
    Bunch : `gammapy.extern.bunch.Bunch`
        Bunch of result maps.   
    """
 
    # Kernel is modified later make a copy here
    kernel = deepcopy(kernel)

    if not kernel.is_bool:
        log.warn('Using weighted kernels can lead to biased results.')

    kernel.normalize('peak')
    counts_ = _convolve_boundary_nan(counts, kernel.array, fft=fft)
    a_ = _convolve_boundary_nan(a_on, kernel.array, fft=fft)
    alpha = a_ / a_off
    background = alpha * off

    significance_lima = significance_on_off(counts_, off, alpha, method='lima')
    
    # safe significance threshold. Is this worse making an option?
    significance_lima[counts_ < 5] = 0

    result = Bunch(significance=significance_lima,
                   counts=counts_,
                   background=background,
                   excess=counts_ - background,
                   alpha=alpha)

    if not exposure is None:
        kernel.normalize('integral')
        exposure_ = _convolve_boundary_nan(exposure, kernel.array, fft=fft)
        flux = (counts_ - background_) / exposure_
        result.flux = flux

    return result

# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" 
Implementation of adaptive smoothing algorithms.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from scipy.signal import fftconvolve
from gammapy.stats import significance
from gammapy.extern.bunch import Bunch
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel
from multiprocessing import Pool, cpu_count
from functools import partial


def _gaussian_cube(data, scales, paralell=True):
    """
    Compute Gaussian scale cube.

    Parameters
    ----------
    
    """
    from scipy.ndimage.filters import gaussian_filter

    # ToDo: check normalization of pi * scale ** 2
    def gaussian_filter_wrap(scale, data):
        return gaussian_filter(data, scale) * np.pi * scale ** 2

    wrap = partial(gaussian_filter_wrap, data=data)
    if parallel:
        pool = Pool()
        results = pool.map(wrap, scales)
        pool.close()
        pool.join()
    else:
        results = map(wrap, scales)
    return np.dstack(results)


def _significance_asmooth(counts, background):
    """
    Significance according to fomula (5) in asmooth paper.
    
    Parameters
    ----------
    counts : ndarray
    	Counts data array
    background : ndarray
    	Background data array.
    """
    return (counts - background) / np.sqrt(counts + background)


def _default_scales(n_scales, factor=np.sqrt(2), kernel=Gaussian2DKernel):
    """
    Create list of Gaussian widths.
    """
    if kernel == Gaussian2DKernel:
        sigma_0 = 1. / np.sqrt(9 * np.pi)
    elif kernel == Tophat2DKernel:
        sigma_0 = 1. / np.sqrt(np.pi)
    return sigma_0 * factor ** np.arange(n_scales)


def adaptive_smooth(skymaps, kernel=Gaussian2DKernel, method='simple', threshold=5,
					scales=None, cache=False):
    """
    Adaptivly smooth counts image, achieving a roughly constant significance
    of features across the whole image.
    
    Algorithm based on http://arxiv.org/pdf/astro-ph/0601306v1.pdf. The
    algorithm was slightly adapted to also allow Li&Ma and TS to estimate the
    signifiance of a feature in the image.
    
    Parameters
    ----------
    counts : ´SkyMap´
        Counts map.
    kernel : astropy.convolution.Kernel
        Smoothing kernel.
    background : ´SkyMap´
    	Backgound map.
    exposure : ´SkyMap´
    	Exposure map.
    scales : list
        Smoothing scales.
    method : str
        Significance estimation method.
    threshold : float
        Significance threshold.
 
    Returns
    -------
    smoothed : Bunch
        Bunch containing the following maps:
            * Smoothed counts
            * Smoothed background
            * Smoothed exposure
            * Smoothed Flux 
            * Scales
            * Significance
            * Smoothed excess
    """
    result = _asmooth_scale_cube()
    
    if cache:
    	with TemporaryFile as f
    		result.write(f)

    scale_cubes = SkyMapCollection.read(f)		
    return _asmooth_scale_cube_to_image(scale_cubes, threshold)


def _asmooth_scale_cube(counts, background, scales, kernel, exposure):
	"""
	"""

    counts_scube = _multiscale_cube(counts, kernel, scales)
    
    if not background is None:
        background_scube = _multiscale_cube(background, kernel, scales)
        background_smoothed = np.tile(np.nan, counts.shape)
    else:
        # Estimate background with ring or adaptive ring
        raise ValueError('Background estimation required.')
    
    if not exposure is None:
        exposure_scube = _multiscale_cube(exposure, kernel, scales)
        exposure_smoothed = np.tile(np.nan, counts.shape)
    
    flux_scube = _multiscale_cube((counts - background) / exposure, kernel, scales)
    flux_smoothed = np.tile(np.nan, counts.shape)
    
    if method == 'lima':
        significance_scube = significance(counts_scube, background_scube, method='lima')
    elif method == 'simple':
        significance_scube = significance(counts_scube, background_scube, method='simple')
    elif method == 'asmooth':
        significance_scube = _significance_asmooth(counts_scube, background_scube)
    elif method == 'ts':
        raise NotImplementedError
    else:
        raise ValueError("Not a valid significance estimation method."
                         " Choose one of the following: 'lima', 'simple', 'asmooth' or 'ts'")
    return result


def _asmooth_scale_cube_to_image(skycubes, threshold):
	"""
	Combine scale cube to image.

	Parameters
	----------
	skycubes : `SkyMapCollection`
		Collection of sky cubes.
	threshold : float
		Significance threshold
	"""
	result = SkyMapCollection()
    counts_smoothed = np.tile(np.nan, counts.shape)
    scale_map = np.tile(np.nan, counts.shape)
    significance_map = np.tile(np.nan, counts.shape)
    
    for i, scale in enumerate(scales):
        mask = (significance_scube[:, :, i] > threshold) & np.isnan(counts_smoothed)
        _ = kernel(scale, mode='oversample')
        _.normalize(mode='peak')
        
        norm = _.array.sum()
        
        counts_smoothed[mask] = counts_scube[:, :, i][mask] / norm
        background_smoothed[mask] = background_scube[:, :, i][mask] / norm
        exposure_smoothed[mask] = exposure_scube[:, :, i][mask] / norm
        flux_smoothed[mask] = flux_scube[:, :, i][mask] / norm
        scale_map[mask] = scale
        significance_map[mask] = significance_scube[:, :, i][mask]
    
    # Set background pixels to largest smoothing scale
    for smoothed, scube in zip([counts_smoothed, background_smoothed, exposure_smoothed, flux_smoothed],
                               [counts_scube, background_scube, exposure_scube, flux_scube]):
        mask = np.isnan(smoothed)
        smoothed[mask] = scube[:, :, i][mask] / norm
    
    return result

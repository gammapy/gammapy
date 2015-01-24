# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions to compute TS maps

"""
from __future__ import print_function, division
import sys
import logging
import warnings
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from astropy.convolution import Tophat2DKernel, Model2DKernel, Gaussian2DKernel
from astropy.convolution.kernels import _round_up_to_odd_integer
from astropy.nddata.utils import extract_array
from astropy.io import fits

from ..stats import cash
from ..irf import multi_gauss_psf_kernel
from ..morphology import Shell2D
from ..extern.zeros import newton
from ..extern.bunch import Bunch
from ..image import (measure_containment_radius, upsample_2N, downsample_2N,
                     shape_2N)


__all__ = ['compute_ts_map', 'compute_ts_map_multiscale',
           'compute_maximum_ts_map', 'TSMapResult']


FLUX_FACTOR = 1E-12
MAX_NITER = 20
CONTAINMENT = 0.8


class TSMapResult(Bunch):
    """
    Represents the TS map computation result.

    Attributes
    ----------
    ts : ndarray
        Estimated TS map
    amplitude : ndarray
        Estimated best fit flux amplitude map
    niter : ndarray
        Number of iterations map
    runtime : float
        Time needed to compute TS map.
    scale : float
        Scale parameter.
    morphology : str
        Source morphology assumption.
    """

    @staticmethod
    def read(filename):
        """
        Read TS map result from file.
        """
        hdu_list = fits.open(filename)
        ts = hdu_list['ts'].data
        amplitude = hdu_list['amplitude'].data
        niter = hdu_list['niter'].data
        scale = hdu_list[0].header['SCALE']
        if scale == 'max':
            scale = hdu_list['scale'].data
        morphology = hdu_list[0].header['MORPH']
        return TSMapResult(ts=ts, amplitude=amplitude, niter=niter,
                           scale=scale, morphology=morphology)

    def write(self, filename, header, overwrite=False):
        """
        Write TS map results to file.
        """
        hdu_list = fits.HDUList()
        if 'MORPH' not in header and hasattr(self, 'morphology'):
            header['MORPH'] = self.morphology, 'Source morphology assumption.'
        if not type(self.scale) == float:
            header['EXTNAME'] = 'scale'
            header['HDUNAME'] = 'scale'
            header['SCALE'] = 'max',  'Source morphology scale parameter.'
            hdu_list.append(fits.ImageHDU(self.scale.astype('float32'), header))
        else:
            header['SCALE'] = self.scale,  'Source morphology scale parameter.'
        for key in ['ts', 'amplitude', 'niter']:
            header['EXTNAME'] = key
            header['HDUNAME'] = key
            hdu_list.append(fits.ImageHDU(self[key].astype('float32'), header))
        hdu_list.writeto(filename, clobber=overwrite)


def f_cash_root(x, counts, background, model):
    """
    Function to find root of. Described in Appendix A, Stewart (2009).

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : array
        Count map slice, where model is defined.
    background : array
        Background map slice, where model is defined.
    model : array
        Source template (multiplied with exposure).
    """
    return (model * (counts / (x * FLUX_FACTOR * model + background) - 1)).sum()


def f_cash(x, counts, background, model):
    """
    Cash fit statistics wrapper for TS map computation.

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : array
        Count map slice, where model is defined.
    background : array
        Background map slice, where model is defined.
    model : array
        Source template (multiplied with exposure).
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return cash(counts, background + x * FLUX_FACTOR * model).sum()


def compute_ts_map_multiscale(maps, psf_parameters, scales=[0], downsample='auto',
                              residual=False, morphology='Gaussian2D', width=None,
                              *args, **kwargs):
    """
    Compute multiscale TS maps using compute_ts_map.

    High level TS map computation using a multi gauss PSF kernel and assuming
    a given source morphology. To optimize the performance the input data
    can be sampled down when computing TS maps on larger scales.

    Parameters
    ----------
    maps : `astropy.io.fits.HDUList`
        HDU list containing the data. The list must contain the following HDU
        extensions:
            * 'On' -- Counts image
            * 'Background' -- Background image
            * 'Diffuse' -- Diffuse model image
            * 'ExpGammaMap' -- Exposure image
    psf_parameters : dict
        Dict defining the multi gauss PSF parameters.
        See `~gammapy.irf.multi_gauss_psf` for details.
    scales : list ([0])
        List of scales to use for TS map computation.
    downsample : int (auto)
        Down sampling factor. Can be set to 'auto' if the down sampling
        factor should be chosen automatically.
    residual : bool (False)
        Compute a TS residual map.
    morphology : str ('Gaussian2D')
        Source morphology assumption. Either 'Gaussian2D' or 'Shell2D'.

    Returns
    -------
    multiscale_result : list
        List of `TSMapResult` objects.
    """
    BINSZ = abs(maps[0].header['CDELT1'])
    shape = maps[0].data.shape
    multiscale_result = []

    for scale in scales:
        logging.info('Computing {0}TS map for scale {1:.3f} deg and {2}'
                     ' morphology.'.format('residual' if residual else '',
                                           scale, morphology))

        # Sample down and require that scale parameters is at least 5 pix
        if downsample == 'auto':
            factor = int(np.select([scale < 5 * BINSZ, scale < 10 * BINSZ,
                                    scale < 20 * BINSZ, scale < 40 * BINSZ],
                                   [1, 2, 4, 8], 16))
        else:
            factor = int(downsample)
        if factor == 1:
            logging.info('No down sampling used.')
            downsampled = False
        else:
            if morphology == 'Shell2D':
                factor /= 2
            logging.info('Using down sampling factor of {0}'.format(factor))
            downsampled = True

        funcs = [np.nansum, np.mean, np.nansum, np.nansum, np.nansum]
        maps_ = {}
        for map_, func in zip(maps, funcs):
            if downsampled:
                maps_[map_.name] = downsample_2N(map_.data, factor, func,
                                                 shape=shape_2N(shape))
            else:
                maps_[map_.name] = map_.data

        # Set up PSF and source kernel
        kernel = multi_gauss_psf_kernel(psf_parameters, BINSZ=BINSZ,
                                        NEW_BINSZ=BINSZ * factor,
                                        mode='oversample')

        if scale > 0:
            from astropy.convolution import convolve
            sigma = scale / (BINSZ * factor)
            if morphology == 'Gaussian2D':
                source_kernel = Gaussian2DKernel(sigma, mode='oversample')
            elif morphology == 'Shell2D':
                model = Shell2D(1, 0, 0, sigma, sigma * width)
                x_size = _round_up_to_odd_integer(2 * sigma * (1 + width)
                                                  + kernel.shape[0] / 2)
                source_kernel = Model2DKernel(model, x_size=x_size, mode='oversample')
            kernel = convolve(source_kernel, kernel)
            kernel.normalize()

        # Compute TS map
        if residual:
            background = (maps_['Background'] + maps_['DIFFUSE'] + maps_['OnModel'])
        else:
            background = maps_['Background'] + maps_['DIFFUSE']
        ts_results = compute_ts_map(maps_['On'], background, maps_['ExpGammaMap'],
                                    kernel, *args, **kwargs)
        logging.info('TS map computation took {0:.1f} s \n'.format(ts_results.runtime))
        ts_results['scale'] = scale
        ts_results['morphology'] = morphology
        if downsampled:
            for name, order in zip(['ts', 'amplitude', 'niter'], [1, 1, 0]):
                ts_results[name] = upsample_2N(ts_results[name], factor,
                                               order=order, shape=shape)
        multiscale_result.append(ts_results)
    return multiscale_result


def compute_maximum_ts_map(ts_map_results):
    """
    Compute maximum TS map across a list of given `TSMapResult` objects.

    Parameters
    ----------
    ts_map_results : list
        List of `TSMapResult` objects.

    Returns
    -------
    TS : `TSMapResult`
        `TSMapResult` object.
    """

    # Get data
    ts = np.dstack([result.ts for result in ts_map_results])
    niter = np.dstack([result.niter for result in ts_map_results])
    amplitude = np.dstack([result.amplitude for result in ts_map_results])
    scales = [result.scale for result in ts_map_results]

    # Set up max arrays
    ts_max = np.max(ts, axis=2)
    scale_max = np.zeros(ts.shape[:-1])
    niter_max = np.zeros(ts.shape[:-1])
    amplitude_max = np.zeros(ts.shape[:-1])

    for i, scale in enumerate(scales):
        index = np.where(ts[:, :, i] == ts_max)
        scale_max[index] = scale
        niter_max[index] = niter[:, :, i][index]
        amplitude_max[index] = amplitude[:, :, i][index]
    return TSMapResult(ts=ts_max, niter=niter_max, amplitude=amplitude_max,
                       morphology=ts_map_results[0].morphology, scale=scale_max)


def compute_ts_map(counts, background, exposure, kernel, mask=None, flux=None,
                   method='root brentq', optimizer='Brent', parallel=True,
                   threshold=None):
    """
    Compute TS map using different optimization methods.

    Parameters
    ----------
    counts : array
        Count map
    background : array
        Background map
    exposure : array
        Exposure map
    kernel : `astropy.convolution.Kernel2D`
        Source model kernel.
    flux : float (None)
        Flux map used as a starting value for the amplitude fit.
    method : str ('root')
        The following options are available:
            * ``'root'`` (default)
                Fit amplitude finding roots of the the derivative of
                the fit statistics. Described in Appendix A in Stewart (2009).
            * ``'fit scipy'``
                Use `scipy.optimize.minimize_scalar` for fitting.
            * ``'fit minuit'``
                Use minuit for fitting.
    optimizer : str ('Brent')
        Which optimizing algorithm to use from scipy. See
        `scipy.optimize.minimize_scalar` for options.
    parallel : bool (True)
        Whether to use multiple cores for parallel processing.
    threshold : float (None)
        If the TS value corresponding to the initial flux estimate is not above
        this threshold, the optimizing step is omitted to save computing time.

    Returns
    -------
    TS : `TSMapResult`
        `TSMapResult` object.

    References
    ----------
    [Stewart2009]_
    """
    from scipy.ndimage.morphology import binary_erosion
    from time import time
    t_0 = time()

    assert counts.shape == background.shape
    assert counts.shape == exposure.shape

    if flux is None and method != 'root brentq':
        from scipy.ndimage import convolve
        radius = _flux_correlation_radius(kernel)
        tophat = Tophat2DKernel(radius, mode='oversample') * np.pi * radius ** 2
        logging.info('Using correlation radius of {0:.1f} pix to estimate'
                     ' initial flux.'.format(radius))
        with np.errstate(invalid='ignore', divide='ignore'):
            flux = (counts - background) / exposure / FLUX_FACTOR
        flux = convolve(flux, tophat.array) / CONTAINMENT

    TS = np.zeros(counts.shape)

    x_min, x_max = kernel.shape[1] // 2, counts.shape[1] - kernel.shape[1] // 2
    y_min, y_max = kernel.shape[0] // 2, counts.shape[0] - kernel.shape[0] // 2
    positions = product(range(y_min, y_max), range(x_min, x_max))

    # Positions where exposure == 0 are not processed
    if mask is None:
        mask = binary_erosion(exposure > 0, np.ones(np.array(kernel.shape) + 1))
    positions = [(j, i) for j, i in positions if mask[j][i]]

    wrap = partial(_ts_value, counts=counts, exposure=exposure,
                   background=background, kernel=kernel, flux=flux,
                   method=method, optimizer=optimizer, threshold=threshold)

    if parallel:
        logging.info('Using {0} cores to compute TS map.'.format(cpu_count()))
        pool = Pool()
        results = pool.map(wrap, positions)
        pool.close()
        pool.join()
    else:
        results = map(wrap, positions)

    # Set TS values at given positions
    j, i = zip(*positions)
    amplitudes = np.zeros(TS.shape)
    niter = np.zeros(TS.shape)
    TS[j, i] = [_[0] for _ in results]
    amplitudes[j, i] = [_[1] for _ in results]
    niter[j, i] = [_[2] for _ in results]
    return TSMapResult(ts=TS, amplitude=amplitudes,
                       niter=niter, runtime=np.round(time() - t_0, 2))


def _ts_value(position, counts, exposure, background, kernel, flux,
              method, optimizer, threshold):
    """
    Compute TS value at a given pixel position i, j using the approach described
    in Stewart (2009).

    Parameters
    ----------
    position : tuple (i, j)
        Pixel position.
    counts : array
        Count map.
    background : array
        Background map.
    exposure : array
        Exposure map.
    kernel : `astropy.convolution.Kernel2D`
        Source model kernel.
    flux : array
        Flux map. The flux value at the given pixel position is used as
        starting value for the minimization.

    Returns
    -------
    TS : float
        TS value at the given pixel position.
    """
    # Get data slices
    counts_ = extract_array(counts, kernel.shape, position).astype(float)
    background_ = extract_array(background, kernel.shape, position).astype(float)
    exposure_ = extract_array(exposure, kernel.shape, position).astype(float)
    model = (exposure_ * kernel._array).astype(float)

    with np.errstate(invalid='ignore', divide='ignore'):
        C_0 = cash(counts_, background_).sum()

    if threshold is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            C_1 = f_cash(flux[position], counts_, background_, model)
        # Don't fit if pixel is low significant
        if C_0 - C_1 < threshold:
            return C_0 - C_1, flux[position] * FLUX_FACTOR, 0

    if method == 'fit minuit':
        amplitude, niter = _fit_amplitude_minuit(counts_, background_, model,
                                                 flux[position])
    elif method == 'fit scipy':
        amplitude, niter = _fit_amplitude_scipy(counts_, background_, model)
    elif method == 'root newton':
        amplitude, niter = _root_amplitude(counts_, background_, model,
                                           flux[position])
    elif method == 'root brentq':
        amplitude, niter = _root_amplitude_brentq(counts_, background_, model)
    else:
        raise ValueError('Invalid fitting method.')
    if niter > MAX_NITER:
        logging.warn('Exceeded maximum number of function evaluations!')
        return np.nan, amplitude, niter

    with np.errstate(invalid='ignore', divide='ignore'):
        C_1 = f_cash(amplitude, counts_, background_, model)

    # Compute and return TS value
    return C_0 - C_1, amplitude * FLUX_FACTOR, niter


def _amplitude_bounds(counts, background, model):
    """
    Compute bounds for     the root of `f_cash_root`.

    Parameters
    ----------
    counts : array
        Count map.
    background : array
        Background map.
    model : array
        Source template (multiplied with exposure).
    """
    # Check that counts slice contains at least one count
    # if not, set flux estimate to -min(B/S).
    with np.errstate(invalid='ignore', divide='ignore'):
        sn = background / model
    sn[counts == 0] = np.inf
    index = np.unravel_index(np.nanargmin(sn), sn.shape)
    bounds = np.array([counts[index], counts.sum()])
    return (bounds / model.sum() - np.nanmin(sn)) / FLUX_FACTOR


def _root_amplitude(counts, background, model, flux):
    """
    Fit amplitude by finding roots using newton algorithm.
    See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : array
        Slice of count map.
    background : array
        Slice of background map.
    model : array
        Model template to fit.
    flux : float
        Starting value for the fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    args = (counts, background, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return newton(f_cash_root, flux, args=args,
                          maxiter=MAX_NITER, tol=0.1)
        except RuntimeError:
            # Where the root finding fails NaN is set as amplitude
            return np.nan, MAX_NITER


def _root_amplitude_brentq(counts, background, model):
    """
    Fit amplitude by finding roots using Brent algorithm.
    See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : array
        Slice of count map.
    background : array
        Slice of background map.
    model : array
        Model template to fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    from scipy.optimize import brentq

    # Compute amplitude bounds and assert counts > 0
    amplitude_min, amplitude_max = _amplitude_bounds(counts, background, model)
    if not counts.sum() > 0:
        return amplitude_min, 0

    args = (counts, background, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = brentq(f_cash_root, amplitude_min, amplitude_max, args=args,
                            maxiter=MAX_NITER, full_output=True, rtol=1E-2)
            return result[0], result[1].iterations
        except (RuntimeError, ValueError):
            # Where the root finding fails NaN is set as amplitude
            return np.nan, MAX_NITER


def _fit_amplitude_scipy(counts, background, model, optimizer='Brent'):
    """
    Fit amplitude using scipy.optimize.

    Parameters
    ----------
    counts : array
        Slice of count map.
    background : array
        Slice of background map.
    model : array
        Model template to fit.
    flux : float
        Starting value for the fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    from scipy.optimize import minimize_scalar
    args = (counts, background, model)
    amplitude_min, amplitude_max = _amplitude_bounds(counts, background, model)
    try:
        result = minimize_scalar(f_cash, bracket=(amplitude_min, amplitude_max),
                                 args=args, method=optimizer, tol=10)
        return result.x, result.nfev
    except ValueError:
        result = minimize_scalar(f_cash, args=args, method=optimizer, tol=0.1)
        return result.x, result.nfev


def _fit_amplitude_minuit(counts, background, model, flux):
    """
    Fit amplitude using minuit.

    Parameters
    ----------
    counts : array
        Slice of count map.
    background : array
        Slice of background map.
    model : array
        Model template to fit.
    flux : float
        Starting value for the fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    from iminuit import Minuit

    def stat(x):
        return f_cash(x, counts, background, model)
    minuit = Minuit(f_cash, x=flux, pedantic=False, print_level=0)
    minuit.migrad()
    return minuit.values['x'], minuit.ncalls


def _flux_correlation_radius(kernel, containment=CONTAINMENT):
    """
    Compute equivalent top-hat kernel radius for a given kernel instance and
    containment fraction.

    Parameters
    ----------
    kernel : `astropy.convolution.Kernel2D`
        Astropy kernel instance.
    containment : float (default = 0.8)
        Containment fraction.

    Returns
    -------
    kernel : float
        Equivalent Tophat kernel radius.
    """
    kernel_image = fits.ImageHDU(kernel.array)
    y, x = kernel.center
    r_c = measure_containment_radius(kernel_image, x, y, containment)
    # Containment radius of Tophat kernel is given by r_c_tophat = r_0 * sqrt(C)
    # by setting r_c = r_c_tophat we can estimate the equivalent containment radius r_0
    return r_c / np.sqrt(containment)


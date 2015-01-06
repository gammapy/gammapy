# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions to compute TS maps

This is in the exploratory phase, we are trying to get a fast tool for a large map.
Here we compare different ways to split the map into parts and different optimizers.
"""
from __future__ import print_function, division
import logging
import warnings
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count
from time import time
import numpy as np
from astropy.io.fits import ImageHDU
from astropy.convolution import Tophat2DKernel
from astropy.nddata.utils import extract_array
from ..stats import cash
from ..image import measure_containment_radius
from ..extern.zeros import newton
from ..extern.bunch import Bunch

__all__ = ['compute_ts_map', 'TSMapResult']


FLUX_FACTOR = 1E-12
MAX_NITER = 20
CONTAINMENT = 0.8


class TSMapResult(Bunch):
    """
    Represents the TS map computation result.

    Attributes
    ----------
    ts : `~numpy.ndarray`
        Estimated TS map
    amplitude : `~numpy.ndarray`
        Estimated best fit flux amplitude map
    niter : `~numpy.ndarray`
        Number of iterations map
    runtime : float
        Time needed to compute TS map.
    """


def f_cash_root(x, counts, background, model):
    """
    Function to find root of. Described in Appendix A, Stewart (2009).

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count map slice, where model is defined.
    background : `~numpy.ndarray`
        Background map slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    return (model - (counts * model) / (background + x * FLUX_FACTOR * model)).sum()


def f_cash(x, counts, background, model):
    """
    Cash fit statistics wrapper for TS map computation.

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count map slice, where model is defined.
    background : `~numpy.ndarray`
        Background map slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return cash(counts, background + x * FLUX_FACTOR * model).sum()


def compute_ts_map(counts, background, exposure, kernel, flux=None,
                   method='root', optimizer='Brent', parallel=True,
                   threshold=None):
    """
    Compute TS map using different fitting methods. 

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map
    background : `~numpy.ndarray`
        Background map
    exposure : `~numpy.ndarray`
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
    ts_map_result : `TSMapResult`
        Result object.

    References
    ----------
    [Stewart2009]_
    """
    from scipy.ndimage.morphology import binary_erosion
    t_0 = time()

    assert counts.shape == background.shape
    assert counts.shape == exposure.shape

    if flux is None:
        from scipy.ndimage import convolve
        radius = _flux_correlation_radius(kernel)
        tophat = Tophat2DKernel(radius, mode='oversample') * np.pi * radius ** 2
        logging.info('Using correlation radius of {0:.1f} pix to estimate initial flux.'.format(radius))
        flux = (counts - background) / exposure / FLUX_FACTOR
        flux = convolve(flux, tophat.array) / CONTAINMENT
    else:
        assert counts.shape == flux.shape

    ts = np.zeros(counts.shape)

    x_min, x_max = kernel.shape[1] // 2, counts.shape[1] - kernel.shape[1] // 2
    y_min, y_max = kernel.shape[0] // 2, counts.shape[0] - kernel.shape[0] // 2
    positions = product(range(x_min, x_max), range(y_min, y_max))

    # Positions where exposure == 0 and flux < 0 are not processed
    mask = binary_erosion(exposure > 0, np.ones(kernel.shape))
    positions = [(i, j) for i, j in positions if mask[j][i] and flux[j][i] > 0]
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
    i, j = zip(*positions)
    amplitudes = np.zeros_like(ts)
    niter = np.zeros_like(ts)
    ts[j, i] = [_[0] for _ in results]
    amplitudes[j, i] = [_[1] for _ in results]
    niter[j, i] = [_[2] for _ in results]

    runtime = time() - t_0

    return TSMapResult(ts=ts, amplitude=amplitudes * FLUX_FACTOR,
                       niter=niter, runtime=runtime)


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
    counts_slice = extract_array(counts, kernel.shape, position).astype(float)
    background_slice = extract_array(background, kernel.shape, position).astype(float)
    exposure_slice = extract_array(exposure, kernel.shape, position).astype(float)
    model = (exposure_slice * kernel._array).astype(float)

    # Compute null hypothesis statistics
    flux_value = flux[position[1]][position[0]]

    with np.errstate(invalid='ignore', divide='ignore'):
        C_0 = cash(counts_slice, background_slice).sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        C_1 = cash(counts_slice, background_slice + flux_value * FLUX_FACTOR * model).sum()

    # Don't fit if pixel is low significant
    TS = C_0 - C_1
    if threshold is not None and TS < threshold:
        return TS, flux_value, 0
    else:
        if method == 'fit minuit':
            amplitude, niter = _fit_amplitude_minuit(counts_slice, background_slice,
                                                     model, flux_value)
        elif method == 'fit scipy':
            amplitude, niter = _fit_amplitude_scipy(counts_slice, background_slice,
                                                    model, flux_value)
        elif method == 'root':
            amplitude, niter = _root_amplitude(counts_slice, background_slice,
                                               model, flux_value)
        if niter > MAX_NITER:
            logging.warning('Exceeded maximum number of function evaluations!')
            return np.nan, amplitude, niter

        with np.errstate(invalid='ignore', divide='ignore'):
            C_1 = cash(counts_slice, background_slice + amplitude * FLUX_FACTOR * model).sum()

        return C_0 - C_1, amplitude, niter


def _root_amplitude(counts, background, model, flux):
    """
    Fit amplitude by finding roots. See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : `~numpy.array`
        Slice of count map.
    background : `~numpy.array`
        Slice of background map.
    model : `~numpy.array`
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
            return newton(f_cash_root, flux, args=args, maxiter=MAX_NITER, tol=0.1)
        except RuntimeError:
            # Where the root finding fails NaN is set as amplitude
            return np.nan, MAX_NITER


def _fit_amplitude_scipy(counts, background, model, flux, optimizer='Brent'):
    """
    Fit amplitude using scipy.optimize.

    Parameters
    ----------
    counts : `~numpy.array`
        Slice of count map.
    background : `~numpy.array`
        Slice of background map.
    model : `~numpy.array`
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
    try:
        bracket = (0, flux, 10 * flux)
        result = minimize_scalar(f_cash, bracket=bracket, args=args,
                                 method=optimizer, tol=10)
        return result.x, result.nfev
    except ValueError:
        result = minimize_scalar(f_cash, args=args, method=optimizer, tol=0.1)
        return result.x, result.nfev


def _fit_amplitude_minuit(counts, background, model, flux):
    """
    Fit amplitude using minuit.

    Parameters
    ----------
    counts : `~numpy.array`
        Slice of count map.
    background : `~numpy.array`
        Slice of background map.
    model : `~numpy.array`
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
        Name of the kernel type.
    containment : float
        Containment fraction.

    Returns
    -------
    kernel : `astropy.convolution.Tophat2DKernel`
        Equivalent top-hat kernel.
    """
    kernel_image = ImageHDU(kernel.array)
    y, x = kernel.center
    r_c = measure_containment_radius(kernel_image, x, y, containment)
    # Containment radius of top-hat kernel is given by r_c_tophat = r_0 * sqrt(C)
    # by setting r_c = r_c_tophat we can estimate the equivalent containment radius r_0
    return r_c / np.sqrt(containment)

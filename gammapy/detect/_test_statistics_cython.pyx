# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

cdef np.float_t FLUX_FACTOR = 1e-12


cdef extern from "math.h":
    float log(float x)

@cython.cdivision(True)
@cython.boundscheck(False)
def _f_cash_root_cython(np.float_t x, np.ndarray[np.float_t, ndim=2] counts,
                        np.ndarray[np.float_t, ndim=2] background,
                        np.ndarray[np.float_t, ndim=2] model):
    """Function to find root of. Described in Appendix A, Stewart (2009).

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count image slice, where model is defined.
    background : `~numpy.ndarray`
        Background image slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    cdef np.float_t sum = 0
    cdef unsigned int i, j, ni, nj
    ni = counts.shape[1]
    nj = counts.shape[0]
    for j in range(nj):
        for i in range(ni):
            if model[j, i] > 0:
                sum += model[j, i] * (1 - counts[j, i] / (x * model[j, i]
                                                          * FLUX_FACTOR + background[j, i]))

    # 2 * FLUX_FACTOR is required to maintain the correct normalization of the
    # derivative of the likelihood function. It doesn't change the result of
    # the fit.
    return 2 * FLUX_FACTOR * sum

@cython.cdivision(True)
@cython.boundscheck(False)
def _x_best_leastsq(np.ndarray[np.float_t, ndim=2] counts,
                    np.ndarray[np.float_t, ndim=2] background,
                    np.ndarray[np.float_t, ndim=2] model,
                    np.ndarray[np.float_t, ndim=2] weights):
    """Best fit amplitude using weighted least squares fit.

    For a single parameter amplitude fit this can be solved analytically.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts image
    background : `~numpy.ndarray`
        Background image
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    weights : `~numpy.ndarray`
        Fit weights.
    """
    cdef np.float_t sum = 0
    cdef np.float_t norm = 0
    cdef unsigned int i, j, ni, nj
    ni = counts.shape[1]
    nj = counts.shape[0]
    for j in range(nj):
        for i in range(ni):
            if model[j, i] > 0 and weights[j, i] > 0:
                sum += (counts[j, i] - background[j, i]) * model[j, i] / weights[j, i]
                norm += model[j, i] * model[j, i] / weights[j, i]
    return sum / norm

@cython.cdivision(True)
@cython.boundscheck(False)
def _amplitude_bounds_cython(np.ndarray[np.float_t, ndim=2] counts,
                             np.ndarray[np.float_t, ndim=2] background,
                             np.ndarray[np.float_t, ndim=2] model):
    """Compute bounds for the root of `_f_cash_root_cython`.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts image
    background : `~numpy.ndarray`
        Background image
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    cdef np.float_t s_model = 0, s_counts = 0, sn, sn_min = 1e14, c_min = 1
    cdef np.float_t b_min, b_max, sn_min_total = 1e14
    cdef unsigned int i, j, ni, nj
    ni = counts.shape[1]
    nj = counts.shape[0]
    for j in range(nj):
        for i in range(ni):
            if counts[j, i] > 0:
                s_counts += counts[j, i]
                if model[j, i] > 0:
                    sn = background[j, i] / model[j, i]
                    if sn < sn_min:
                        sn_min = sn
                        c_min = counts[j, i]
            if model[j, i] > 0:
                s_model += model[j, i]
                sn = background[j, i] / model[j, i]
                if sn < sn_min_total:
                    sn_min_total = sn
    b_min = c_min / s_model - sn_min
    b_max = s_counts / s_model - sn_min
    return b_min / FLUX_FACTOR, b_max / FLUX_FACTOR, -sn_min_total / FLUX_FACTOR

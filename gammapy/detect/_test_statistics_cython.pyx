# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
cimport numpy as np
cimport cython

cdef np.float_t FLUX_FACTOR = 1E-12


@cython.cdivision(True)
@cython.boundscheck(False)
def _f_cash_root_cython(np.float_t x, np.ndarray[np.float_t, ndim=2] counts,
                        np.ndarray[np.float_t, ndim=2] background,
                        np.ndarray[np.float_t, ndim=2] model):
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
    cdef np.float_t sum
    cdef unsigned int i, j, ni, nj
    ni = counts.shape[1]
    nj = counts.shape[0]
    sum = 0
    for j in range(nj):
        for i in range(ni):
            if model[j, i] > 0:
                sum += (model[j, i] * (counts[j, i] / (x * FLUX_FACTOR * model[j, i]
                                                       + background[j, i]) - 1))
    return sum


@cython.cdivision(True)
@cython.boundscheck(False)
def _x_best_leastsq(np.ndarray[np.float_t, ndim=2] counts,
                    np.ndarray[np.float_t, ndim=2] background,
                    np.ndarray[np.float_t, ndim=2] model,
                    np.ndarray[np.float_t, ndim=2] weights):
    """
    Best fit amplitude using weighted least squares fit.

    For a single parameter amplitude fit this can be solved analytically.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map.
    background : `~numpy.ndarray`
        Background map.
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
    """
    Compute bounds for the root of `_f_cash_root_cython`.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map.
    background : `~numpy.ndarray`
        Background map.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """

    cdef np.float_t s_model = 0, s_counts = 0, sn, sn_min = 1E14, c_min = 1
    cdef np.float_t b_min, b_max
    cdef unsigned int i, j, ni, nj
    ni = counts.shape[1]
    nj = counts.shape[0]
    for j in range(ni):
        for i in range(nj):
            s_model += model[j, i]
            if counts[j, i] > 0:
                s_counts += counts[j, i]
                if model[j, i] > 0:
                    sn = background[j, i] / model[j, i]
                    if sn < sn_min:
                        sn_min = sn
                        c_min = counts[j, i]
    b_min = c_min / s_model - sn_min
    b_max = s_counts / s_model - sn_min
    return b_min / FLUX_FACTOR, b_max / FLUX_FACTOR


IF UNAME_SYSNAME == "Windows":
    # Windows fall back for log2, which was only introduced in MSVC 2010
    cdef extern from "math.h":
        float log(float x)

    cdef np.float_t LOG2_TO_E = 1.

    def log2(x):
        return log(x)
ELSE:
    cdef extern from "math.h":
        float log2(float x)

    cdef np.float_t LOG2_TO_E = 0.69314718055994529


@cython.cdivision(True)
@cython.boundscheck(False)
def _cash_cython(np.ndarray[np.float_t, ndim=2] counts,
                 np.ndarray[np.float_t, ndim=2] model):
    """
    Cash fit statistics.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """

    cdef unsigned int i, j, ni, nj
    ni = counts.shape[1]
    nj = counts.shape[0]
    cdef np.ndarray[np.float_t, ndim=2] cash = np.zeros([nj, ni], dtype=float)

    for j in range(nj):
        for i in range(ni):
            if model[j, i] > 0:
                cash[j, i] = 2 * (model[j, i] - counts[j, i] * log2(model[j, i]) * LOG2_TO_E)
    return cash

@cython.cdivision(True)
@cython.boundscheck(False)
def _cash_sum_cython(np.ndarray[np.float_t, ndim=2] counts,
                     np.ndarray[np.float_t, ndim=2] model):
    """
    Summed cash fit statistics.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map slice, where model is defined.
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
                sum += model[j, i] - counts[j, i] * log2(model[j, i]) * LOG2_TO_E
    return 2 * sum


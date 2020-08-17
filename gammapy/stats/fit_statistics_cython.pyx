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
def cash_sum_cython(np.ndarray[np.float_t, ndim=1] counts,
                    np.ndarray[np.float_t, ndim=1] npred):
    """Summed cash fit statistics.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts array.
    npred : `~numpy.ndarray`
        Predicted counts array.
    """
    cdef np.float_t sum = 0
    cdef unsigned int i, ni
    ni = counts.shape[0]
    for i in range(ni):
        if npred[i] > 0:
            sum += npred[i]
            if counts[i] > 0:
                sum -= counts[i] * log(npred[i])
    return 2 * sum


@cython.cdivision(True)
@cython.boundscheck(False)
def f_cash_root_cython(np.float_t x, np.ndarray[np.float_t, ndim=1] counts,
                       np.ndarray[np.float_t, ndim=1] background,
                       np.ndarray[np.float_t, ndim=1] model):
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
    cdef unsigned int i, ni
    ni = counts.shape[0]
    for i in range(ni):
        if model[i] > 0:
            sum += model[i] * (1 - counts[i] / (x * model[i]
                                                      * FLUX_FACTOR + background[i]))

    # 2 * FLUX_FACTOR is required to maintain the correct normalization of the
    # derivative of the likelihood function. It doesn't change the result of
    # the fit.
    return 2 * FLUX_FACTOR * sum


@cython.cdivision(True)
@cython.boundscheck(False)
def amplitude_bounds_cython(np.ndarray[np.float_t, ndim=1] counts,
                            np.ndarray[np.float_t, ndim=1] background,
                            np.ndarray[np.float_t, ndim=1] model):
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
    cdef unsigned int i, ni
    ni = counts.shape[0]
    for i in range(ni):
        if counts[i] > 0:
            s_counts += counts[i]
            if model[i] > 0:
                sn = background[i] / model[i]
                if sn < sn_min:
                    sn_min = sn
                    c_min = counts[i]
        if model[i] > 0:
            s_model += model[i]
            sn = background[i] / model[i]
            if sn < sn_min_total:
                sn_min_total = sn
    b_min = c_min / s_model - sn_min
    b_max = s_counts / s_model - sn_min
    return b_min / FLUX_FACTOR, b_max / FLUX_FACTOR, -sn_min_total / FLUX_FACTOR

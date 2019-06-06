# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

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
def cstat_sum_cython(np.ndarray[np.float_t, ndim=1] counts,
                     np.ndarray[np.float_t, ndim=1] npred):
    """Summed cstat fit statistics.

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
                sum += (- counts[i] + counts[i] * log(counts[i] / npred[i]))
    return 2 * sum
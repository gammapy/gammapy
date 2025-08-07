# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numba import jit

global TRUNCATION_VALUE  # compile time constant
TRUNCATION_VALUE = 1e-25


@jit("f8(f8[:],f8[:],f8[:])", nopython=True, nogil=True, cache=True)
def weighted_cash_sum_jit(counts, npred, weight):
    """Cash fit statistics with weights.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts array.
    npred : `~numpy.ndarray`
        Predicted counts array.
    weight : `~numpy.ndarray`
        likelihood weights array.
    """
    stat_sum = 0.0
    trunc = TRUNCATION_VALUE
    logtrunc = np.log(TRUNCATION_VALUE)

    ni = counts.shape[0]
    for i in range(ni):
        npr = npred[i]
        if npr > trunc:
            lognpr = np.log(npr)
        else:
            npr = trunc
            lognpr = logtrunc

        if weight[i] > 0:
            stat_sum += weight[i] * npr
            if counts[i] > 0.0:
                stat_sum -= weight[i] * counts[i] * lognpr

    return 2 * stat_sum


@jit("f8(f8[:],f8[:])", nopython=True, nogil=True, cache=True)
def cash_sum_jit(counts, npred):
    """Sum cash fit statistics.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts array.
    npred : `~numpy.ndarray`
        Predicted counts array.
    """
    stat_sum = 0.0
    trunc = TRUNCATION_VALUE
    logtrunc = np.log(TRUNCATION_VALUE)

    ni = counts.shape[0]
    for i in range(ni):
        npr = npred[i]
        if npr > trunc:
            lognpr = np.log(npr)
        else:
            npr = trunc
            lognpr = logtrunc

        stat_sum += npr
        if counts[i] > 0.0:
            stat_sum -= counts[i] * lognpr

    return 2 * stat_sum


@jit("f8(f8,f8[:],f8[:],f8[:])", nopython=True, nogil=True, cache=True)
def f_cash_root_jit(x, counts, background, model):
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
    stat_sum = 0.0
    ni = counts.shape[0]
    for i in range(ni):
        if model[i] > 0.0:
            if counts[i] > 0.0:
                denom = x * model[i] + background[i]
                if denom != 0.0:
                    stat_sum += model[i] * (1.0 - counts[i] / denom)
            else:
                stat_sum += model[i]

    # 2 is required to maintain the correct normalization of the
    # derivative of the likelihood function. It doesn't change the result of
    # the fit.
    return 2 * stat_sum


@jit("UniTuple(f8, 3)(f8[:],f8[:],f8[:])", nopython=True, nogil=True, cache=True)
def norm_bounds_jit(counts, background, model):
    """Compute bounds for the root of `_f_cash_root_jit`.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts image
    background : `~numpy.ndarray`
        Background image
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    s_model = 0.0
    s_counts = 0.0
    sn_min = 1e14
    c_min = 1.0
    sn_min_total = 1e14
    ni = counts.shape[0]
    for i in range(ni):
        if counts[i] > 0.0:
            s_counts += counts[i]
            if model[i] > 0.0:
                sn = background[i] / model[i]
                if sn < sn_min:
                    sn_min = sn
                    c_min = counts[i]
        if model[i] > 0.0:
            s_model += model[i]
            sn = background[i] / model[i]
            if sn < sn_min_total:
                sn_min_total = sn
    if s_model == 0.0:
        b_min = np.nan
        b_max = np.nan
    else:
        b_min = c_min / s_model - sn_min
        b_max = s_counts / s_model - sn_min
    return b_min, b_max, -sn_min_total

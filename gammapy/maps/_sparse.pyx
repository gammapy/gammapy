# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

ctypedef fused int_t:
    np.int32_t
    np.int64_t

ctypedef fused float_t:
    np.float32_t
    np.float64_t

@cython.cdivision(True)
@cython.boundscheck(False)
def binary_search(np.ndarray[int_t, ndim=1] arr_in, int_t idx,
                  int_t ilo, int_t ihi):
    """Perform a binary search for an element ``idx`` in a sorted integer
    array ``arr_in``.  The search is restricted to the range between
    ``ilo`` and ``ihi``.

    Parameters
    ----------
    arr_in : `~numpy.ndarray`
        Sorted array of integers.
    idx : int
        Value to be searched for in ``arr_in``.

    Returns
    -------
    idx_out: int
        Index of matching element or the index at which ``idx`` would
        be inserted to maintain sorted order.
    """
    cdef int_t mid = 0
    cdef int_t midval = 0

    while ilo < ihi:
        mid = (ilo + ihi) // 2
        midval = arr_in[mid]

        if midval < idx:
            ilo = mid + 1
        elif midval > idx:
            ihi = mid
        else:
            return mid

    return ilo

@cython.cdivision(True)
@cython.boundscheck(False)
def find_in_array(np.ndarray[int_t, ndim=1] idx0,
                  np.ndarray[int_t, ndim=1] idx1):
    """Find the values in ``idx0`` contained in ``idx1``.

    Parameters
    ----------
    idx0 : `~numpy.ndarray`
        Array of unsorted input values.
    idx1 : `~numpy.ndarray`
        Array of sorted values to be searched. 

    Returns
    -------
    idx : `~numpy.ndarray`
        Array of indices with length of ``idx0`` with the position of
        the given element in ``idx1``.
    msk : `~numpy.ndarray`
        Boolean mask with length of ``idx0`` indicating whether a
        given value was found in ``idx1``.
    """
    cdef int ni = idx0.shape[0]
    cdef int nj = idx1.shape[0]

    cdef np.ndarray[int_t, ndim = 1] idx_sort = np.argsort(idx0)
    cdef int i = 0
    cdef int ii = 0
    cdef int_t j = binary_search(idx1, idx0[idx_sort[0]], 0, nj)
    cdef np.ndarray[int_t, ndim= 1] out = np.zeros([ni], dtype=idx0.dtype)
    cdef np.ndarray[np.uint8_t, ndim= 1] msk = np.zeros([ni], dtype=np.uint8)

    while i < ni and j < nj:

        ii = idx_sort[i]

        # Current element is less
        if idx0[ii] < idx1[j]:
            out[ii] = j
            i += 1
        elif idx0[ii] > idx1[j]:
            j += 1
        else:
            out[ii] = j
            msk[ii] = True
            i += 1

    return out, msk.astype(bool)

@cython.cdivision(True)
@cython.boundscheck(False)
def merge_sparse_arrays(np.ndarray[int_t, ndim=1] idx0,
                        np.ndarray[float_t, ndim=1] val0,
                        np.ndarray[int_t, ndim=1] idx1,
                        np.ndarray[float_t, ndim=1] val1,
                        bint fill=False
                        ):
    """Merge two sparse arrays represented as index/value pairs into a
    single sparse array.

    Values in the first array (``idx0``, ``val0``) will supersede those
    in the second array. Indices in the second array should be presorted.

    Parameters
    ----------
    idx0 : `~numpy.ndarray`
        Array of indices of first sparse array.
    val0 : `~numpy.ndarray`
        Array of values of first sparse array.
    idx1 : `~numpy.ndarray`
        Array of indices for second sparse array.
    val1 : `~numpy.ndarray`
        Array of values for second sparse array.
    fill : bool
        Flag to switch between update and fill mode.  When fill is
        True the values in the first array will be added to the second
        array.  When fill is False values in the second array will be
        updated to the values in the first array.

    Returns
    -------
    idx : `~numpy.ndarray`
        Array of indices for merged sparse array.
    vals : `~numpy.ndarray`
        Array of values for merged sparse array.
    """
    cdef int ni = idx0.shape[0]
    cdef int nj = idx1.shape[0]
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    isort = np.argsort(idx0)
    idx0 = idx0[isort]
    val0 = val0[isort]

    cdef np.ndarray[int_t, ndim = 1] idx = np.sort(np.unique(np.concatenate((idx0, idx1))))
    cdef int n = idx.size
    cdef np.ndarray[float_t, ndim= 1] vals = np.zeros(n, dtype=val0.dtype)

    while k < n:

        while j < nj:

            if idx1[j] > idx[k]:
                break
            elif idx1[j] == idx[k]:

                if fill:
                    vals[k] += val1[j]
                else:
                    vals[k] = val1[j]

            j += 1

        while i < ni:

            if idx0[i] > idx[k]:
                break
            elif idx0[i] == idx[k]:

                if fill:
                    vals[k] += val0[i]
                else:
                    vals[k] = val0[i]

            i += 1

        k += 1

    return idx, vals

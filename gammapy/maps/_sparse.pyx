# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
cimport numpy as np
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
def binary_search(np.ndarray[np.int64_t, ndim=1] arr_in, np.int64_t idx,
                  np.int64_t ilo, np.int64_t ihi):
    """Perform a binary search for an element ``idx`` in a sorted integer
    array ``arr_in``.  The search is restricted to the range between
    ``ilo`` and ``ihi``.  This will return the index of the matching
    element or the index at which ``idx`` would be inserted to
    maintain sorted order.

    Parameters
    ----------
    arr_in : `~numpy.ndarray`
        Sorted array of integers.

    idx : int
        Value to be searched for.

    """
    
    cdef int mid = 0
    cdef int midval = 0

    while(ilo < ihi):
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
def binary_search2(np.ndarray[np.int64_t, ndim=1] arr_in, np.int64_t idx,
                   np.int64_t ilo, np.int64_t ihi):

    cdef int mid = 0
    cdef int midval = 0

    while(ilo < ihi):
        mid = (ilo + ihi) // 2
        midval = arr_in[mid]

        if midval < idx:
            ilo = mid + 1
        elif midval > idx:
            ihi = mid
        else:
            return mid, 1

    return ilo, 0


@cython.cdivision(True)
@cython.boundscheck(False)
def find_in_array(np.ndarray[np.int64_t, ndim=1] idx0,
                  np.ndarray[np.int64_t, ndim=1] idx1):
    """Find the values in ``idx0`` contained in ``idx1``.

    Parameters
    ----------
    idx0 : `~np.ndarray`
        Array of unsorted input values.

    idx1 : `~np.ndarray`
        Array of sorted values to be searched. 

    Returns
    -------
    idx : `~np.ndarray`
        Array of indices with length of ``idx0`` with the position of
        the given element in ``idx1``.

    msk : `~np.ndarray`
        Boolean mask with length of ``idx0`` indicating whether a
        given value was found in ``idx1``.

    """

    cdef int ni = idx0.shape[0]
    cdef int nj = idx1.shape[0]

    cdef np.ndarray[np.int64_t, ndim= 1] idx_sort = np.argsort(idx0)
    cdef int i = 0
    cdef int ii = 0
    cdef np.int64_t j = binary_search(idx1, idx0[idx_sort[0]], 0, nj)
    cdef np.ndarray[np.int64_t, ndim = 1] out = np.zeros([ni], dtype=np.int64)
    cdef np.ndarray[np.uint8_t, ndim = 1] msk = np.zeros([ni], dtype=np.uint8)

    while(i < ni and j < nj):

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
def merge_sparse_arrays(np.ndarray[np.int64_t, ndim=1] idx0,
                        np.ndarray[np.float64_t, ndim=1] val0,
                        np.ndarray[np.int64_t, ndim=1] idx1,
                        np.ndarray[np.float64_t, ndim=1] val1,
                        np.int64_t fill=False
                        ):
    """Merge two sparse arrays represented as index/value pairs into a
    single sparse array.  Values in the first array (``idx0``,
    ``val0``) will supersede those in the second array.  Indices in
    the second array should be presorted.

    Parameters
    ----------
    idx0 : `numpy.ndarray`
        Array of indices of first sparse array.

    val0 : `numpy.ndarray`
        Array of values of first sparse array.

    idx1 : `numpy.ndarray`
        Array of indices for second sparse array. 

    val1 : `numpy.ndarray`
        Array of values for second sparse array. 

    fill : bool
        Flag to switch between update and fill mode.  When fill is
        True the values in the first array will be added to the second
        array.  When fill is False values in the second array will be
        updated to the values in the first array.

    Returns
    -------
    idx : `numpy.ndarray`
        Array of indices for merged sparse array.

    vals : `numpy.ndarray`
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

    cdef np.ndarray[np.int64_t, ndim= 1] idx = np.sort(np.unique(np.concatenate((idx0, idx1))))
    cdef int n = idx.size
    cdef np.ndarray[np.float64_t, ndim = 1] vals = np.zeros(n, dtype=np.float64)
    while(k < n):

        while(j < nj):

            if(idx1[j] > idx[k]):
                break
            elif(idx1[j] == idx[k]):

                if fill:
                    vals[k] += val1[j]
                else:
                    vals[k] = val1[j]

            j += 1

        while(i < ni):

            if(idx0[i] > idx[k]):
                break
            elif(idx0[i] == idx[k]):

                if fill:
                    vals[k] += val0[i]
                else:
                    vals[k] = val0[i]

            i += 1

        k += 1

    return idx, vals

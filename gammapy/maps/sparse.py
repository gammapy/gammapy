# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ._sparse import find_in_array, merge_sparse_arrays

__all__ = ["SparseArray"]


def slices_to_idxs(slices, shape, ndim):
    if slices == Ellipsis:
        slices = tuple([slice(None)] * ndim)
    elif not isinstance(slices, tuple):
        slices = tuple([slices])

    if len(slices) < ndim:
        slices = tuple(list(slices) + [slice(None)] * (ndim - len(slices)))

    # nslice = min(1, sum([not isinstance(s, slice) for s in slices]))
    # nslice += sum([isinstance(s, slice) for s in slices])
    nslice = len(slices)
    idim = None
    idx = []
    for i, s in enumerate(slices):

        si = [None] * nslice
        if isinstance(s, slice):
            si[i] = slice(None)
            start = 0 if s.start is None else s.start
            stop = shape[i] if s.stop is None else s.stop

            idx += [np.arange(start, stop, 1)[tuple(si)]]
        else:
            if idim is None:
                idim = i
            si[idim] = slice(None)
            idx += [np.array(s, ndmin=1)[tuple(si)]]

    return idx


class SparseArray(object):
    """Sparse N-dimensional array object.

    This class implements a data structure for sparse n-dimensional
    arrays such that only non-zero data values are allocated in memory.
    Supports numpy conventions for indexing and slicing logic.

    Parameters
    ----------
    shape : tuple of ints
        Shape of array.
    idx : `~numpy.ndarray`, optional
        Flattened index vector that initializes the array.  If none
        then an empty array will be created.
    data : `~numpy.ndarray`, optional
        Flattened data vector that initializes the array.  If none
        then an empty array will be created.
    dtype : data-type, optional
        Type of data vector.
    fill_value : scalar, optional
        Value assigned to array elements that are not allocated in
        memory.

    Examples
    --------
    A SparseArray is created in the same way as `~numpy.ndarray` by
    passing the array shape to the constructor with an optional
    argument for the array type:

    >>> import numpy as np
    >>> from gammapy.maps import SparseArray
    >>> v = SparseArray((10,20), dtype=float)

    Alternatively you can create a new SparseArray from an
    `~numpy.ndarray` with `~SparseArray.from_array`:

    >>> x = np.ones((10,20))
    >>> v = SparseArray.from_array(x)

    SparseArray follows numpy indexing and slicing conventions for
    setting/getting array elements.  The primary difference with
    respect to the behavior of `~numpy.ndarray` is that indexing
    always returns a copy rather than a view.

    >>> v[0,0] = 1.0
    >>> print(v[0,0])
    >>> v[:,0] = 1.0
    >>> print(v[:,0])
    """

    def __init__(self, shape, idx=None, data=None, dtype=float, fill_value=0.0):
        self._shape = tuple(shape)
        self._fill_value = fill_value
        if idx is not None:
            self._idx = idx
            self._data = data
        else:
            self._idx = np.zeros(0, dtype=np.int64)
            self._data = np.zeros(0, dtype=dtype)

    def __getitem__(self, slices):
        idx = slices_to_idxs(slices, self.shape, self.ndim)
        idx = np.broadcast_arrays(*idx)
        return self.get(idx)

    def __setitem__(self, slices, vals):
        idx = slices_to_idxs(slices, self.shape, self.ndim)
        idx = np.broadcast_arrays(*idx)
        return self.set(idx, vals)

    @property
    def size(self):
        """Return current number of elements."""
        return len(self._data)

    @property
    def data(self):
        """Return the sparsified data array."""
        return self._data

    @property
    def dtype(self):
        """Return the type of the data array member."""
        return self._data.dtype

    @property
    def idx(self):
        """Return flattened index vector."""
        return self._idx

    @property
    def shape(self):
        """Array shape."""
        return self._shape

    @property
    def ndim(self):
        """Array number of dimensions (int)."""
        return len(self._shape)

    @classmethod
    def from_array(cls, data, min_value=0.0):
        """Create a `~SparseArray` from a numpy array.

        Parameters
        ----------
        data : `numpy.ndarray`
            Input data array.
        min_value : float
            Threshold for sparsifying the data vector.

        Returns
        -------
        out : `~SparseArray`
            Output sparse array.
        """
        shape = data.shape
        idx = np.where(data > min_value)
        idx = np.ravel_multi_index(idx, shape)
        data = data[data > min_value]
        return cls(shape, idx, data)

    def _to_flat_index(self, idx_in):
        """Convert index tuple to a flattened index."""
        idx_in = tuple([np.array(z, ndmin=1, copy=False) for z in idx_in])
        msk = np.all(np.stack([t < n for t, n in zip(idx_in, self.shape)]), axis=0)
        idx = np.ravel_multi_index(
            tuple([t[msk] for t in idx_in]), self.shape, mode="wrap"
        )

        return idx, msk

    def set(self, idx_in, vals, fill=False):
        """Set array values at indices ``idx_in``."""
        o = np.broadcast_arrays(vals, *idx_in)
        vals = np.ravel(o[0])

        # TODO: Determine whether new vs. existing indices are being
        # addressed, in the latter case we only need to update data
        # array

        vals = np.array(vals, ndmin=1)
        idx_flat_in, msk_in = self._to_flat_index(idx_in)
        vals = np.asanyarray(vals, dtype=self.data.dtype)
        idx, data = merge_sparse_arrays(
            idx_flat_in, vals, self.idx, self.data, fill=fill
        )

        # Remove elements equal to fill value
        msk = data != self._fill_value
        idx = idx[msk]
        data = data[msk]
        self._idx = idx
        self._data = data
        # idx, msk = find_in_array(idx_flat_in, self.idx)
        # self._data[idx[msk]] = vals[msk]

    def get(self, idx_in):
        """Get array values at indices ``idx_in``."""
        shape_out = idx_in[0].shape
        idx_flat_in, msk_in = self._to_flat_index(idx_in)
        idx, msk = find_in_array(idx_flat_in, self.idx)
        val_out = np.full(shape_out, self._fill_value)
        val_out.flat[np.flatnonzero(msk_in)[msk]] = self._data[idx[msk]]
        return np.squeeze(val_out)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, **unused_kwargs):
        # FIXME: Figure out how to correctly support np.apply_over_axes

        if axis is None:
            return np.sum(self._data)
        else:
            shape = list(self.shape)
            if keepdims:
                shape[axis] = 1
            else:
                del shape[axis]
            out = SparseArray(shape)
            return out

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..sparse import SparseArray
from .._sparse import merge_sparse_arrays

test_params = [(8,), (8, 16), (8, 16, 32)]


@pytest.mark.parametrize("shape", test_params)
def test_sparse_init(shape):
    v = SparseArray(shape)
    assert v.shape == shape
    assert v.size == 0

    data = np.ones(shape)
    v = SparseArray.from_array(data)
    assert_allclose(v[...], data)


def test_sparse_getitem():
    shape = (8, 16, 32)

    data = np.random.poisson(np.ones(shape)).astype(float)
    v = SparseArray.from_array(data)
    assert_allclose(v[...], data[...])
    assert_allclose(v[:], data[:])
    assert_allclose(v[:, :], data[:, :])
    assert_allclose(v[:, :, :], data[:, :, :])
    assert_allclose(v[1, 3, 10], data[1, 3, 10])
    assert_allclose(v[:, 3, 10], data[:, 3, 10])
    assert_allclose(v[1, :, 10], data[1, :, 10])
    assert_allclose(v[1, 3, :], data[1, 3, :])
    assert_allclose(v[1, np.arange(4), :], data[1, np.arange(4), :])
    assert_allclose(
        v[np.arange(4), np.arange(4), :], data[np.arange(4), np.arange(4), :]
    )
    assert_allclose(
        v[:, np.arange(4), np.arange(4)], data[:, np.arange(4), np.arange(4)]
    )


@pytest.mark.parametrize("shape", test_params)
def test_sparse_setitem(shape):
    data = np.random.poisson(np.ones(shape)).astype(float)
    v = SparseArray(shape)
    idx = np.where(data > 0)
    v[idx] = data[idx]
    assert_allclose(v[...], data)

    v = SparseArray(shape)
    v.set(idx, data[idx])
    assert_allclose(v[...], data)

    v = SparseArray(shape)
    v.set(idx, data[idx], fill=True)
    v.set(idx, data[idx], fill=True)
    assert_allclose(v[...], 2.0 * data)

    v = SparseArray(shape)
    data0 = np.random.poisson(np.ones(shape)).astype(float)
    data1 = np.random.poisson(np.ones(shape)).astype(float)
    data = data0 + data1
    idx0 = np.where(data0 > 0)
    idx1 = np.where(data1 > 0)
    idx_in = tuple([np.concatenate((x, y)) for x, y in zip(idx0, idx1)])
    data_in = np.concatenate((data0[idx0], data1[idx1]))
    v.set(idx_in, data_in, fill=True)
    assert_allclose(v[...], data)


@pytest.mark.parametrize(
    ("dtype_idx", "dtype_val"),
    [
        (np.int64, np.float64),
        (np.int32, np.float64),
        (np.int32, np.float32),
        (np.int32, np.float64),
    ],
)
def test_merge_sparse_arrays(dtype_idx, dtype_val):
    idx0 = np.array([0, 0, 1, 4], dtype=dtype_idx)
    val0 = np.array([1.0, 2.0, 3.0, 7.0], dtype=dtype_val)
    idx1 = np.array([0, 1, 2], dtype=dtype_idx)
    val1 = np.array([1.0, 1.0, 1.0], dtype=dtype_val)
    idx, val = merge_sparse_arrays(idx0, val0, idx1, val1)
    assert_allclose(idx, np.unique(np.concatenate((idx0, idx1))))
    assert_allclose(val, np.array([2.0, 3.0, 1.0, 7.0]))
    idx, val = merge_sparse_arrays(idx0, val0, idx1, val1, True)
    assert_allclose(idx, np.unique(np.concatenate((idx0, idx1))))
    assert_allclose(val, np.array([4.0, 4.0, 1.0, 7.0]))

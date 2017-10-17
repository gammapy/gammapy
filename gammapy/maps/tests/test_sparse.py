# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..sparse import SparseArray

test_params = [
    (8,),
    (8, 16),
    (8, 16, 32),
]


@pytest.mark.parametrize(('shape'), test_params)
def test_sparse_init(shape):

    v = SparseArray(shape)
    assert(v.shape == shape)
    assert(v.size == 0)

    data = np.ones(shape)
    v = SparseArray.from_array(data)
    assert_allclose(v[...], data)


def test_sparse_getitem():

    shape = (8, 16, 32)

    data = np.random.poisson(np.ones(shape)).astype(float)
    v = SparseArray.from_array(data)
    assert_allclose(v[...], data[...])
    assert_allclose(v[:, :, :], data[:, :, :])
    assert_allclose(v[1, 3, 10], data[1, 3, 10])
    assert_allclose(v[:, 3, 10], data[:, 3, 10])
    assert_allclose(v[1, :, 10], data[1, :, 10])
    assert_allclose(v[1, 3, :], data[1, 3, :])
    assert_allclose(v[1, np.arange(4), :], data[1, np.arange(4), :])
    assert_allclose(v[np.arange(4), np.arange(4), :],
                    data[np.arange(4), np.arange(4), :])
    assert_allclose(v[:, np.arange(4), np.arange(4)],
                    data[:, np.arange(4), np.arange(4)])


@pytest.mark.parametrize(('shape'), test_params)
def test_sparse_setitem(shape):

    data = np.random.poisson(np.ones(shape)).astype(float)
    v = SparseArray(shape)
    idx = np.where(data > 0)
    v[idx] = data[idx]
    assert_allclose(v[...], data)

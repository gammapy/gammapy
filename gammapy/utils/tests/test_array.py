# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.utils.array import array_stats_str, shape_2N, shape_divisible_by


def test_array_stats_str():
    actual = array_stats_str(np.pi, "pi")
    assert actual == "pi             : size =     1, min =  3.142, max =  3.142\n"

    actual = array_stats_str([np.pi, 42])
    assert actual == "size =     2, min =  3.142, max = 42.000\n"


def test_shape_2N():
    shape = (34, 89, 120, 444)
    expected_shape = (40, 96, 128, 448)
    assert expected_shape == shape_2N(shape=shape, N=3)


def test_shape_divisible_by():
    shape = (7, 8, 10, 13)
    new_shape = shape_divisible_by(shape=shape, factor=4)
    assert new_shape == (8, 8, 12, 16)

    # shapes that are already divisible are left unchanged
    assert shape_divisible_by(shape=(12, 16), factor=4) == (12, 16)

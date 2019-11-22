# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.utils.interpolation import LogScale
from gammapy.utils.testing import assert_allclose


def test_LogScale_inverse():
    tiny = np.finfo(np.float32).tiny
    log_scale = LogScale()
    values = np.array([1, 1e-5, 1e-40])
    log_values = log_scale(values)
    assert_allclose(log_values, np.array([0, np.log(1e-5), np.log(tiny)]))
    inv_values = log_scale.inverse(log_values)
    assert_allclose(inv_values, np.array([1, 1e-5, 0]))

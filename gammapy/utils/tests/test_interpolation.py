# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.utils.interpolation import LogScale
from gammapy.utils.testing import assert_allclose


def test_logscale_adaptive_behavior():
    """
    Test that LogScale adapts its clipping threshold based on input dtype.
    """
    log_scale = LogScale()

    # Case 1: float64 input (High Precision)
    # -----------------------------------------------------------
    # 1e-40 is well within float64 limits (~1e-308), so it should NOT be clipped.
    values_64 = np.array([1, 1e-5, 1e-40], dtype=np.float64)
    log_values_64 = log_scale(values_64)

    # Expectation: actual log(1e-40), not log(1e-38)
    # This verifies the fix for the "Silent Bias" issue.
    assert_allclose(log_values_64, np.log(values_64))

    # Inverse check: Should recover 1e-40 exactly, not 0.
    inv_values_64 = log_scale.inverse(log_values_64)
    assert_allclose(inv_values_64, values_64)

    # Case 2: float32 input (Standard Precision)
    # -----------------------------------------------------------
    # 0 is smaller than float32 tiny (~1.17e-38), so it SHOULD be clipped
    # to maintain numerical stability for 32-bit workflows.
    tiny_32 = np.finfo(np.float32).tiny
    values_32 = np.array([1, 1e-5, 0], dtype=np.float32)

    log_values_32 = log_scale(values_32)

    # Expectation: 0 is clipped to tiny_32
    expected_clipped_32 = np.array([1, 1e-5, tiny_32], dtype=np.float32)
    assert_allclose(log_values_32, np.log(expected_clipped_32))

    # Inverse check: Under the original LogScale semantics, values close to tiny
    # are mapped back to 0.
    inv_values_32 = log_scale.inverse(log_values_32)
    expected_inv_32 = np.array([1, 1e-5, 0], dtype=np.float32)
    assert_allclose(inv_values_32, expected_inv_32, rtol=1e-6)

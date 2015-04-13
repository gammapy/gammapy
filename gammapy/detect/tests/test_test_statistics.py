# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np
from numpy.testing.utils import assert_allclose

from astropy.tests.helper import pytest
from astropy.convolution import Gaussian2DKernel


from ...detect import compute_ts_map
from ...datasets import load_poisson_stats_image
from ...image.utils import upsample_2N, downsample_2N

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_compute_ts_map():
    """Minimal test of compute_ts_map"""
    data = load_poisson_stats_image(extra_info=True)
    kernel = Gaussian2DKernel(2.5)
    data['exposure'] = np.ones(data['counts'].shape) * 1E12
    for _, func in zip(['counts', 'background', 'exposure'], [np.nansum, np.nansum, np.mean]):
        data[_] = downsample_2N(data[_], 2, func)

    result = compute_ts_map(data['counts'], data['background'], data['exposure'],
                            kernel)
    for name, order in zip(['ts', 'amplitude', 'niter'], [2, 5, 0]):
        result[name] = np.nan_to_num(result[name])
        result[name] = upsample_2N(result[name], 2, order=order)

    assert_allclose(1705.840212274973, result.ts[99, 99], rtol=1e-3)
    assert_allclose([[99], [99]], np.where(result.ts == result.ts.max()))
    assert_allclose(6, result.niter[99, 99])
    assert_allclose(1.0227934338735763e-09, result.amplitude[99, 99], rtol=1e-3)

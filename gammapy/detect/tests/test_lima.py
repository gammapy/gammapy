# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing.utils import assert_allclose, assert_equal

from astropy.convolution import Tophat2DKernel

from ...utils.testing import requires_dependency, requires_data
from ...detect import compute_ts_map, compute_lima_map
from ...datasets import load_poisson_stats_image


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_compute_lima_map():
    """
    Test Li&Ma map against TS map for Tophat kernel
    """
    data = load_poisson_stats_image(extra_info=True)

    kernel = Tophat2DKernel(5)
    result_lima = compute_lima_map(data['counts'], data['background'], kernel,
                                   data['exposure'])
    kernel.normalize('integral')
    result_ts = compute_ts_map(data['counts'], data['background'], data['exposure'],
                            kernel)
    
    # Ignore NaN values by setting them to zero 
    assert_allclose(np.nan_to_num(result_ts.sqrt_ts),
                    np.nan_to_num(result_lima.significance), atol=1E-3)
    assert_allclose(np.nan_to_num(result_ts.amplitude),
                    np.nan_to_num(result_lima.flux), atol=3E-12)


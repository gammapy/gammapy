# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing.utils import assert_allclose, assert_equal

from astropy.convolution import Tophat2DKernel
from astropy.io import fits

from ...utils.testing import requires_dependency, requires_data
from ...detect import compute_ts_map, compute_lima_map, compute_lima_on_off_map
from ...datasets.core import _GammapyExtra
from ...datasets import load_poisson_stats_image
from ...data import FitsMapBunch

from pathlib import Path


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
    
    assert_allclose(result_ts.sqrt_ts, result_lima.significance, atol=1E-3)
    assert_allclose(result_ts.amplitude, result_lima.flux, atol=3E-12)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_compute_lima_on_off_map():
    """
    Test Li&Ma map with snippet from the H.E.S.S. survey data.
    """
    filename = _GammapyExtra().filename('test_datasets/unbundled/'
                                        'hess/survey/hess_survey_snippet.fits.gz')
    data = FitsMapBunch.read(filename)

    kernel = Tophat2DKernel(5)

    result_lima = compute_lima_on_off_map(data.On, data.Off, data.OnExposure,
                                          data.OffExposure, kernel)
    
    # Set boundary to NaN in reference image
    data.Significance[np.isnan(result_lima.significance)] = np.nan
    assert_allclose(result_lima.significance, data.Significance, atol=1E-5)

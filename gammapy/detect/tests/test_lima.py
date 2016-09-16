# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing.utils import assert_allclose
from astropy.convolution import Tophat2DKernel
from ...utils.testing import requires_dependency, requires_data
from ...detect import compute_ts_image, compute_lima_image, compute_lima_on_off_image
from ...datasets import load_poisson_stats_image, gammapy_extra
from ...image import SkyImageCollection, SkyImage


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_compute_lima_image():
    """
    Test Li&Ma image against TS image for Tophat kernel
    """
    filenames = load_poisson_stats_image(extra_info=True, return_filenames=True)
    data = SkyImageCollection()
    data.counts = SkyImage.read(filenames['counts'])
    data.background = SkyImage.read(filenames['background'])
    data.exposure = SkyImage.read(filenames['exposure'])

    kernel = Tophat2DKernel(5)
    result_lima = compute_lima_image(data['counts'], data['background'], kernel,
                                     data['exposure'])
    kernel.normalize('integral')
    result_ts = compute_ts_image(data['counts'], data['background'], data['exposure'],
                                 kernel)

    assert_allclose(result_ts.sqrt_ts, result_lima.significance, atol=1E-3)
    assert_allclose(result_ts.amplitude, result_lima.flux, atol=3E-12)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_compute_lima_on_off_image():
    """
    Test Li&Ma image with snippet from the H.E.S.S. survey data.
    """
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/survey/'
                                      'hess_survey_snippet.fits.gz')
    maps = SkyImageCollection.read(filename)

    kernel = Tophat2DKernel(5)

    result_lima = compute_lima_on_off_image(maps.on.data, maps.off.data, maps.onexposure.data,
                                            maps.offexposure.data, kernel)

    # Reproduce safe significance threshold from HESS software
    result_lima.significance.data[result_lima.n_on.data < 5] = 0

    # Set boundary to NaN in reference image
    maps.significance.data[np.isnan(result_lima.significance)] = np.nan
    assert_allclose(result_lima.significance, maps.significance, atol=1E-5)

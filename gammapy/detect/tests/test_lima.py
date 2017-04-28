# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing.utils import assert_allclose
from astropy.convolution import Tophat2DKernel
from ...utils.testing import requires_dependency, requires_data
from ...detect import compute_ts_image, compute_lima_image, compute_lima_on_off_image
from ...image import SkyImageList


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_compute_lima_image():
    """
    Test Li & Ma image against TS image for Tophat kernel
    """
    filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz'
    images = SkyImageList.read(filename)

    kernel = Tophat2DKernel(5)
    result_lima = compute_lima_image(
        images['counts'], images['background'], kernel, images['exposure'],
    )
    kernel.normalize('integral')
    result_ts = compute_ts_image(
        images['counts'], images['background'], images['exposure'], kernel,
    )

    assert_allclose(result_ts['sqrt_ts'], result_lima['significance'], atol=1e-3)
    assert_allclose(result_ts['amplitude'], result_lima['flux'], atol=3e-12)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_compute_lima_on_off_image():
    """
    Test Li & Ma image with snippet from the H.E.S.S. survey data.
    """
    filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/hess/survey/hess_survey_snippet.fits.gz'
    images = SkyImageList.read(filename)

    kernel = Tophat2DKernel(5)

    results = compute_lima_on_off_image(
        images['On'].data, images['Off'].data,
        images['OnExposure'].data, images['OffExposure'].data,
        kernel,
    )

    # Reproduce safe significance threshold from HESS software
    results['significance'].data[results['n_on'].data < 5] = 0

    # Set boundary to NaN in reference image
    images['Significance'].data[np.isnan(results['significance'])] = np.nan
    assert_allclose(results['significance'], images['Significance'], atol=1e-5)

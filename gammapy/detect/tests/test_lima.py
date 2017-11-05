# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing.utils import assert_allclose
from astropy.convolution import Tophat2DKernel
from ...utils.testing import requires_dependency, requires_data
from ...detect import TSImageEstimator, compute_lima_image, compute_lima_on_off_image
from ...image import SkyImage, SkyImageList


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_compute_lima_image():
    """
    Test Li & Ma image against TS image for Tophat kernel
    """
    filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz'
    counts = SkyImage.read(filename, hdu='counts')
    background = SkyImage.read(filename, hdu='background')
    exposure = SkyImage.read(filename, hdu='exposure')

    kernel = Tophat2DKernel(5)
    result_lima = compute_lima_image(
        counts, background, kernel, exposure,
    )

    kernel.normalize('integral')
    ts_estimator = TSImageEstimator()
    images = SkyImageList([counts, background, exposure])
    result_ts = ts_estimator.run(images, kernel)

    assert_allclose(result_ts['sqrt_ts'], result_lima['significance'], atol=1e-3)
    assert_allclose(result_ts['flux'], result_lima['flux'], atol=3e-12)


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
        images['On'], images['Off'],
        images['OnExposure'], images['OffExposure'],
        kernel,
    )

    # Reproduce safe significance threshold from HESS software
    results['significance'].data[results['n_on'].data < 5] = 0

    # Set boundary to NaN in reference image
    s = images['Significance'].data.copy()
    s[np.isnan(results['significance'])] = np.nan
    assert_allclose(results['significance'], s, atol=1e-5)

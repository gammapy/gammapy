# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.convolution import Gaussian2DKernel
from ...detect import matched_filter

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_center():

    # Test dataset parameters
    x_size, y_size = (11, 11)
    total_excess = 100
    total_background = 1000
    ones = np.ones((x_size, y_size))

    kernel = Gaussian2DKernel(3, x_size=x_size, y_size=y_size).array
    excess = total_excess * ones / ones.sum()
    background = total_background * ones / ones.sum()
    counts = excess + background
    images = dict(counts=counts, background=background)

    probability = matched_filter.probability_center(images, kernel)
    # TODO: try to get a verified result
    assert_allclose(probability, 0.0043809799783148338)

    significance = matched_filter.significance_center(images, kernel)
    # TODO: try to get a verified result
    assert_allclose(significance, 2.6212048333735858)


@pytest.mark.xfail
@pytest.mark.skipif('not HAS_SCIPY')
def test_image():
    # Test dataset parameters
    x_size_kernel, y_size_kernel = (11, 11)
    x_size_image, y_size_image = (31, 31)
    total_excess = 100
    total_background = 1000
    ones = np.ones((x_size_image, y_size_image))

    # Create test dataset
    kernel = Gaussian2DKernel(3, x_size=x_size_kernel, y_size=y_size_kernel).array
    excess = total_excess * Gaussian2DKernel(3, x_size=x_size_image, y_size=y_size_image).array
    background = total_background * ones / ones.sum()
    counts = excess + background
    #np.random.seed(0)
    #counts = np.random.poisson(counts)
    images = dict(counts=counts, background=background)

    probability = matched_filter.probability_image(images, kernel)
    # TODO: try to get a verified result
    assert_allclose(probability.max(), 0.48409238192500076)

    significance = matched_filter.significance_image(images, kernel)
    # TODO: try to get a verified result
    assert_allclose(significance.max(), 7.2493488182450569)

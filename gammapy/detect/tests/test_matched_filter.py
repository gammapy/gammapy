# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from astropy.convolution import Gaussian2DKernel
from .. import matched_filter

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

    kernel = Gaussian2DKernel(width=3, x_size=x_size, y_size=y_size).array
    excess = total_excess * ones / ones.sum()
    background = total_background * ones / ones.sum()
    counts = excess + background
    images = dict(counts=counts, background=background)
    
    probability = matched_filter.probability_center(images, kernel)
    # TODO: try to get a verified result
    assert_almost_equal(probability, 0.0043809799783148338)

    significance = matched_filter.significance_center(images, kernel)
    # TODO: try to get a verified result
    assert_almost_equal(significance, 2.6212048333735858)


@pytest.mark.skipif('not HAS_SCIPY')
def test_image():
    # Test dataset parameters
    x_size, y_size = (11, 11)
    image_shape = (31, 31)
    total_excess = 100
    total_background = 1000
    ones = np.ones((x_size, y_size))
    
    # Create test dataset
    kernel = Gaussian2DKernel(width=3, x_size=x_size, y_size=y_size).array
    excess = total_excess * kernel
    background = total_background * ones / ones.sum()
    counts = excess + background
    #np.random.seed(0)
    #counts = np.random.poisson(counts)
    images = dict(counts=counts, background=background)

    probability = matched_filter.probability_image(images, kernel)
    # TODO: try to get a verified result
    assert_almost_equal(probability.max(), 0.16047699425893236)

    significance = matched_filter.significance_image(images, kernel)
    # TODO: try to get a verified result
    assert_almost_equal(significance.max(), 3.1325069098248197)

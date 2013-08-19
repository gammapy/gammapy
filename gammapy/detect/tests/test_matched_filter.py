# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_almost_equal
import pytest
from astropy.nddata import make_kernel
from .. import matched_filter

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_center():
    # Test dataset parameters
    kernel_shape = (11, 11)
    total_excess = 100
    total_background = 1000

    kernel = make_kernel(kernel_shape, kernelwidth=3, kerneltype='gaussian')
    excess = total_excess * make_kernel(kernel_shape, kernelwidth=kernel_shape[0], kerneltype='boxcar')
    background = total_background * make_kernel(kernel_shape, kernelwidth=kernel_shape[0], kerneltype='boxcar')
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
    kernel_shape = (11, 11)
    image_shape = (31, 31)
    total_excess = 100
    total_background = 1000
    
    # Create test dataset
    kernel = make_kernel(kernel_shape, kernelwidth=3, kerneltype='gaussian')
    excess = total_excess * make_kernel(image_shape, kernelwidth=3, kerneltype='gaussian')
    background = total_background * make_kernel(image_shape, kernelwidth=image_shape[0], kerneltype='boxcar')
    counts = excess + background
    #np.random.seed(0)
    #counts = np.random.poisson(counts)
    images = dict(counts=counts, background=background)

    probability = matched_filter.probability_image(images, kernel)
    # TODO: try to get a verified result
    assert_almost_equal(probability.max(), 0.4840923814509871)

    significance = matched_filter.significance_image(images, kernel)
    # TODO: try to get a verified result
    assert_almost_equal(significance.max(), 7.249351301729793)

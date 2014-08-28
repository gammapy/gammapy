# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import os
import tempfile
from astropy.io import fits
from astropy.tests.helper import pytest
from ...background import GammaImages, IterativeKernelBackgroundEstimator
from ...image import make_empty_image
from ...stats import significance

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_GammaImages():
    """Tests compute correlated maps in GammaImages.
    This is the only method in GammaImages that actually calculates anything.
    """
    # Set up test counts and background
    counts = make_empty_image(nxpix=10, nypix=10, binsz=1, fill=42)
    counts.data[4][4] = 1000

    background_data = np.ones_like(counts.data, dtype=float)
    background = fits.ImageHDU(data=background_data,
                               header=counts.header)

    # Single unit pixel kernel so should actually be no change.
    background_kernel = np.ones((1, 1))
    
    images = GammaImages(counts, background)
    images.compute_correlated_maps(background_kernel)
    
    # Test significance image against Li & Ma significance value
    expected = significance(counts.data, background.data)
    actual = images.significance

    assert_allclose(actual, expected)


@pytest.mark.skipif('not HAS_SCIPY')
class TestIterativeKernelBackgroundEstimator(object):
    """Tests methods in the IterativeKernelBackgroundEstimator.
    """
    def setup_class(self):
        """Prepares appropriate input and defines inputs for test cases.
        """

        # Load/create example model images
        counts = make_empty_image(nxpix=10, nypix=10, binsz=1, fill=42)
        counts.data[4][4] = 1000
        # Initial counts required by one of the tests.
        self.counts = counts

        # Start with flat background estimate
        # Background must be provided as an ImageHDU
        background_data = np.ones_like(counts.data, dtype=float)
        background = fits.ImageHDU(data=background_data,
                                   header=counts.header)

        images = GammaImages(counts=counts, background=background)

        source_kernel = np.ones((1, 3))
        source_kernel /= source_kernel.sum()

        background_kernel = np.ones((5, 3))
        significance_threshold = 4
        mask_dilation_radius = 0.3

        # Loads prepared inputs into estimator
        
        self.ibe = IterativeKernelBackgroundEstimator(
                                                      images,
                                                      source_kernel,
                                                      background_kernel,
                                                      significance_threshold,
                                                      mask_dilation_radius,
                                                      )

    def test_run_ntimes(self):
        """Show that the mask is stable after first few iterations."""

        mask1, _ = self.ibe.run_ntimes(n_iterations=5)

        mask2, _ = self.ibe.run_ntimes(n_iterations=10)
        assert_allclose(mask1.data.sum(), mask2.data.sum())

        mask3, _ = self.ibe.run_ntimes(n_iterations=15)
        assert_allclose(mask1.data.sum(), mask3.data.sum())
        
    def test_run(self):
        """Asserts that mask and background are as expected according to input."""

        mask, background = self.ibe.run()

        # Check mask matches expectations
        expected_mask = np.ones_like(self.counts.data)
        expected_mask[4][3] = 0
        expected_mask[4][4] = 0
        expected_mask[4][5] = 0

        assert_equal(mask.data, expected_mask)

        # Check background, should be 42 uniformly
        assert_allclose(background.data, 42 * np.ones((10, 10)))
    
    def test_save_files(self):
        """Tests that files are saves, and checks values within them."""
        # Create temporary file to write output into
        dir = tempfile.mkdtemp()
        self.ibe.run_iteration(1)
        self.ibe.save_files(filebase=dir, index=0)

        mask_filename = dir + '00_mask.fits'
        significance_filename = dir + '00_significance.fits'
        background_filename = dir + '00_background.fits'

        mask_data = fits.open(mask_filename)[1].data
        significance_data = fits.open(significance_filename)[1].data
        background_data = fits.open(background_filename)[1].data

        # Checks values in files against known results for one iteration.
        assert_allclose(mask_data.sum(), 97)
        assert_allclose(significance_data.sum(), 90.82654795219804)
        assert_allclose(background_data.sum(), 4200)

        os.removedirs(dir)

    def teardown_class(self):
        pass

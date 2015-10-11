# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.tests.helper import pytest
from astropy.units import Quantity
from astropy.coordinates.angles import Angle
from ...background import GammaImages, IterativeKernelBackgroundEstimator
from ...image import make_empty_image
from ...stats import significance
from ...datasets import FermiGalacticCenter

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
    counts_hdu = make_empty_image(nxpix=10, nypix=10, binsz=1, fill=42)
    counts_hdu.data[4][4] = 1000
    counts = counts_hdu.data

    background_data = 42 * np.ones_like(counts, dtype=float)

    # Single unit pixel kernel so should actually be no change.
    background_kernel = np.ones((1, 1))

    images = GammaImages(counts, background_data)
    images.compute_correlated_maps(background_kernel)

    # Test significance image against Li & Ma significance value
    expected = significance(counts, background_data)
    actual = images.significance

    assert_allclose(actual, expected)


@pytest.mark.skipif('not HAS_SCIPY')
class TestIterativeKernelBackgroundEstimator(object):
    """Tests methods in the IterativeKernelBackgroundEstimator.
    """

    def setup_class(self):
        """Prepares appropriate input and defines inputs for test cases.
        """
        from scipy.ndimage import convolve
        # Load/create example model images
        counts_hdu = make_empty_image(nxpix=10, nypix=10, binsz=1, fill=42)
        counts_hdu.data[4][4] = 1000
        counts = counts_hdu.data
        # Initial counts required by one of the tests.
        self.counts = counts

        psf = FermiGalacticCenter.psf()
        psf = psf.table_psf_in_energy_band(Quantity([10, 500], 'GeV'))
        kernel_array = psf.kernel(pixel_size=Angle(1, 'deg'),
                                  offset_max=Angle(3, 'deg'), normalize=True)

        counts_blob = convolve(counts, kernel_array, mode='constant')

        self.counts_blob = counts_blob

        # Start with flat background estimate
        # Background must be provided as an ImageHDU

        images = GammaImages(counts=counts, header=counts_hdu.header)

        images_blob = GammaImages(counts=counts_blob, header=counts_hdu.header)

        source_kernel = np.ones((1, 3))

        background_kernel = np.ones((5, 3))

        significance_threshold = 4
        mask_dilation_radius = 1

        # Loads prepared inputs into estimator

        self.ibe = IterativeKernelBackgroundEstimator(
            images,
            source_kernel,
            background_kernel,
            significance_threshold,
            mask_dilation_radius
        )

        self.ibe2 = IterativeKernelBackgroundEstimator(
            images,
            source_kernel,
            background_kernel,
            significance_threshold,
            mask_dilation_radius
        )

        self.ibe_blob = IterativeKernelBackgroundEstimator(
            images_blob,
            source_kernel,
            background_kernel,
            significance_threshold,
            mask_dilation_radius
        )

    def test_run_iteration_point(self):
        """Asserts that mask and background are as expected according to input."""

        # Call the run_iteration code as this is what is explicitly being tested
        self.ibe.run_iteration()
        # Should be run twice to update the mask
        self.ibe.run_iteration()
        mask = self.ibe.mask_image_hdu.data
        background = self.ibe.background_image_hdu.data

        # Check mask matches expectations
        expected_mask = np.ones_like(self.counts)
        expected_mask[4][3] = 0
        expected_mask[4][4] = 0
        expected_mask[4][5] = 0

        assert_allclose(mask.astype(int), expected_mask)
        # Check background, should be 42 uniformly
        assert_allclose(background.astype(float), 42 * np.ones((10, 10)))

    def test_run_iteration_blob(self):
        """Asserts that mask and background are as expected according to input."""

        # Call the run_iteration code as this is what is explicitly being tested
        self.ibe_blob.run_iteration()
        # Should be run twice to update the mask
        self.ibe_blob.run_iteration()
        background = self.ibe_blob.background_image_hdu.data
        # Check background, should be 42 uniformly within 10%
        assert_allclose(background, 42 * np.ones((10, 10)), rtol=0.15)

    def test_run(self):
        """Tests run script."""
        mask, background = self.ibe2.run()

        assert_allclose(mask.sum(), 97)
        assert_allclose(background, 42 * np.ones((10, 10)))

    def test_save_files(self, tmpdir):
        """Tests that files are saves, and checks values within them."""
        # Create temporary file to write output into
        self.ibe.run_iteration(1)
        self.ibe.save_files(filebase=str(tmpdir), index=0)

        mask_filename = str(tmpdir.join('00_mask.fits'))
        significance_filename = str(tmpdir.join('00_significance.fits'))
        background_filename = str(tmpdir.join('00_background.fits'))

        mask_data = fits.open(mask_filename)[1].data
        significance_data = fits.open(significance_filename)[1].data
        background_data = fits.open(background_filename)[1].data

        # Checks values in files against known results for one iteration.
        assert_allclose(mask_data.sum(), 97)
        assert_allclose(significance_data.sum(), 157.316195729298)
        assert_allclose(background_data.sum(), 4200)

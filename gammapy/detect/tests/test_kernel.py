# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.units import Quantity
from astropy.coordinates.angles import Angle
from ...utils.testing import requires_dependency, requires_data
from ...image import SkyImage
from ...stats import significance
from ...datasets import FermiGalacticCenter
from ..kernel import KernelBackgroundEstimatorData, KernelBackgroundEstimator


@requires_dependency('scipy')
def test_KernelBackgroundEstimatorData():
    """Tests compute correlated images in KernelBackgroundEstimatorData.
    This is the only method in KernelBackgroundEstimatorData that actually calculates anything.
    """
    # Set up test counts and background
    counts_hdu = SkyImage.empty(nxpix=10, nypix=10, binsz=1, fill=42).to_image_hdu()
    counts_hdu.data[4][4] = 1000
    counts = counts_hdu.data

    background_data = 42 * np.ones_like(counts, dtype=float)

    # Single unit pixel kernel so should actually be no change.
    background_kernel = np.ones((1, 1))

    images = KernelBackgroundEstimatorData(counts, background_data)
    images.compute_correlated_images(background_kernel)

    # Test significance image against Li & Ma significance value
    expected = significance(counts, background_data)
    actual = images.significance

    assert_allclose(actual, expected)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestKernelBackgroundEstimator(object):

    def setup_class(self):
        """Prepares appropriate input and defines inputs for test cases.
        """
        from scipy.ndimage import convolve
        # Load/create example model images
        counts_hdu = SkyImage.empty(nxpix=10, nypix=10, binsz=1, fill=42).to_image_hdu()
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

        images = KernelBackgroundEstimatorData(counts=counts, header=counts_hdu.header)

        images_blob = KernelBackgroundEstimatorData(counts=counts_blob, header=counts_hdu.header)

        source_kernel = np.ones((1, 3))

        background_kernel = np.ones((5, 3))

        significance_threshold = 4
        mask_dilation_radius = 1

        # Loads prepared inputs into estimator

        self.kbe = KernelBackgroundEstimator(
            images,
            source_kernel,
            background_kernel,
            significance_threshold,
            mask_dilation_radius
        )

        self.kbe2 = KernelBackgroundEstimator(
            images,
            source_kernel,
            background_kernel,
            significance_threshold,
            mask_dilation_radius
        )

        self.kbe_blob = KernelBackgroundEstimator(
            images_blob,
            source_kernel,
            background_kernel,
            significance_threshold,
            mask_dilation_radius
        )

    def test_run_iteration_point(self):
        """Asserts that mask and background are as expected according to input."""

        # Call the run_iteration code as this is what is explicitly being tested
        self.kbe.run_iteration()
        # Should be run twice to update the mask
        self.kbe.run_iteration()
        mask = self.kbe.mask_image_hdu.data
        background = self.kbe.background_image_hdu.data

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
        self.kbe_blob.run_iteration()
        # Should be run twice to update the mask
        self.kbe_blob.run_iteration()
        background = self.kbe_blob.background_image_hdu.data
        # Check background, should be 42 uniformly within 10%
        assert_allclose(background, 42 * np.ones((10, 10)), rtol=0.15)

    def test_run(self):
        """Tests run script."""
        mask, background = self.kbe2.run()

        assert_allclose(mask.sum(), 97)
        assert_allclose(background, 42 * np.ones((10, 10)))

    def test_save_files(self, tmpdir):
        """Tests that files are saves, and checks values within them."""
        # Create temporary file to write output into
        self.kbe.run_iteration(1)
        self.kbe.save_files(base_dir=str(tmpdir), index=0)

        filename = tmpdir / '00_mask.fits'
        mask = fits.open(str(filename))[1].data

        filename = tmpdir / '00_significance.fits'
        significance = fits.open(str(filename))[1].data

        filename = tmpdir / '00_background.fits'
        background = fits.open(str(filename))[1].data

        # Checks values in files against known results for one iteration.
        assert_allclose(mask.sum(), 97)
        assert_allclose(significance.sum(), 157.316195729298)
        assert_allclose(background.sum(), 4200)

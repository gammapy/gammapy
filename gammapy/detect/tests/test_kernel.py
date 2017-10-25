# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data
from ...image import SkyImage, SkyImageList
from ...datasets import FermiGalacticCenter
from ..kernel import KernelBackgroundEstimator


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestKernelBackgroundEstimator(object):
    def setup_class(self):
        """Prepares appropriate input and defines inputs for test cases.
        """
        source_kernel = np.ones((1, 3))
        background_kernel = np.ones((5, 3))

        # Loads prepared inputs into estimator
        self.kbe = KernelBackgroundEstimator(
            kernel_src=source_kernel,
            kernel_bkg=background_kernel,
            significance_threshold=4,
            mask_dilation_radius=1 * u.deg
        )

    def _images_point(self):
        """
        Test images for a point source
        """
        counts = SkyImage.empty(name='counts', nxpix=10, nypix=10, binsz=1, fill=42.)
        counts.data[4][4] = 1000

        background = SkyImage.empty_like(counts, fill=42., name='background')
        exclusion = SkyImage.empty_like(counts, name='exclusion', fill=1.)
        return SkyImageList([counts, background, exclusion])

    def _images_psf(self):
        # Initial counts required by one of the tests.
        images = self._images_point()

        psf = FermiGalacticCenter.psf()
        erange = [10, 500] * u.GeV
        psf = psf.table_psf_in_energy_band(erange)
        rad_max = psf.containment_radius(0.99)
        kernel_array = psf.kernel(images['counts'], rad_max=rad_max)

        images['counts'] = images['counts'].convolve(kernel_array, mode='constant')
        return images

    def test_run_iteration_point(self):
        """Asserts that mask and background are as expected according to input."""

        images = self._images_point()

        # Call the _run_iteration code as this is what is explicitly being tested
        result = self.kbe._run_iteration(images)
        mask, background = result['exclusion'].data, result['background'].data

        # Check mask
        idx = [(4, 3), (4, 4), (4, 5), (3, 4), (5, 4)]
        i, j = zip(*idx)
        assert_allclose(mask[i, j], 0)
        assert_allclose((1. - mask).sum(), 11)

        # Check background, should be 42 uniformly
        assert_allclose(background, 42 * np.ones((10, 10)))

    def test_run_iteration_blob(self):
        """Asserts that mask and background are as expected according to input."""

        images = self._images_psf()

        # Call the run_iteration code as this is what is explicitly being tested
        result = self.kbe._run_iteration(images)
        background = result['background'].data

        # Check background, should be 42 uniformly within 10%
        assert_allclose(background, 42 * np.ones((10, 10)), rtol=0.1)

    def test_run(self):
        """Tests run script."""
        images = self._images_point()
        result = self.kbe.run(images)
        mask, background = result['exclusion'].data, result['background'].data

        assert_allclose(mask.sum(), 89)
        assert_allclose(background, 42 * np.ones((10, 10)))

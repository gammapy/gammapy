# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ...utils.testing import requires_data
from ...image import SkyImage, SkyImageList
from ..kernel import KernelBackgroundEstimator

pytest.importorskip('scipy')


@pytest.fixture(scope='session')
def images():
    """A simple test case for the algorithm."""
    counts = SkyImage.empty(name='counts', nxpix=10, nypix=10, binsz=1, fill=42.)
    counts.data[4][4] = 1000

    background = SkyImage.empty_like(counts, fill=42., name='background')
    exclusion = SkyImage.empty_like(counts, name='exclusion', fill=1.)
    return SkyImageList([counts, background, exclusion])


@pytest.fixture()
def kbe():
    source_kernel = np.ones((1, 3))
    background_kernel = np.ones((5, 3))
    return KernelBackgroundEstimator(
        kernel_src=source_kernel,
        kernel_bkg=background_kernel,
        significance_threshold=4,
        mask_dilation_radius='1 deg',
        keep_record=True,
    )


@requires_data('gammapy-extra')
def test_run_iteration(kbe, images):
    # Call the _run_iteration code as this is what is explicitly being tested
    result = kbe._run_iteration(images)
    mask, background = result['exclusion'].data, result['background'].data

    # Check mask
    idx = [(4, 3), (4, 4), (4, 5), (3, 4), (5, 4)]
    i, j = zip(*idx)
    assert_allclose(mask[i, j], 0)
    assert_allclose((1. - mask).sum(), 11)

    # Check background, should be 42 uniformly
    assert_allclose(background, 42 * np.ones((10, 10)))


@requires_data('gammapy-extra')
def test_run(kbe, images):
    result = kbe.run(images)
    mask, background = result['exclusion'].data, result['background'].data

    assert_allclose(mask.sum(), 89)
    assert_allclose(background, 42 * np.ones((10, 10)))
    assert len(kbe.images_stack) == 4

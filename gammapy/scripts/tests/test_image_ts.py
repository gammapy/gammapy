# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing.utils import assert_allclose
import pytest
from ...utils.testing import requires_dependency, requires_data
from ...image import SkyImageList
from ..image_ts import image_ts_main

SCALES = ['0.000', '0.050', '0.100', '0.200']


@requires_dependency('scipy')
@requires_dependency('skimage')
@requires_data('gammapy-extra')
@pytest.mark.parametrize('scale', SCALES)
def test_command_line_gammapy_image_ts(tmpdir, scale):
    """Minimal test of gammapy_image_ts using testcase that
    guaranteed to work with compute_ts_image"""
    input_dir = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/'
    input_filename = input_dir + 'input_all.fits.gz'
    psf_filename = input_dir + 'psf.json'
    expected_filename = input_dir + 'expected_ts_{}.fits.gz'.format(scale)
    actual_filename = str(tmpdir / 'output.fits')

    args = [input_filename, actual_filename,
            "--psf", psf_filename,
            "--scales", scale]
    image_ts_main(args)

    actual = SkyImageList.read(actual_filename)
    expected = SkyImageList.read(expected_filename)

    opts = dict(rtol=1e-2, atol=1e-5, equal_nan=True)
    assert_allclose(actual['ts'].data, expected['ts'].data, **opts)
    assert_allclose(actual['sqrt_ts'].data, expected['sqrt_ts'].data, **opts)
    assert_allclose(actual['flux'].data, expected['amplitude'].data, **opts)

    assert 'niter' in actual.names

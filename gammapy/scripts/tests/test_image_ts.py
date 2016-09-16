# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import json
from numpy.testing.utils import assert_allclose
from astropy.tests.helper import pytest
from astropy.io import fits
from ...utils.testing import requires_dependency, requires_data
from ...image import SkyImageCollection
from ...datasets import load_poisson_stats_image
from ..image_ts import image_ts_main

SCALES = ['0.000', '0.050', '0.100', '0.200']


@requires_dependency('scipy')
@requires_dependency('skimage')
@requires_data('gammapy-extra')
class TestImageTS:
    @pytest.fixture(autouse=True)
    def init_test_input_file(self, tmpdir):
        self.input_filename = str(tmpdir / 'input_all.fits.gz')

        data = load_poisson_stats_image(extra_info=True)
        header = data['header']
        images = [data['counts'], data['background'], data['exposure']]
        image_names = ['Counts', 'Background', 'Exposure']

        hdu_list = fits.HDUList()
        for image, name in zip(images, image_names):
            hdu = fits.ImageHDU(data=image, header=header, name=name)
            hdu_list.append(hdu)

        hdu_list.writeto(self.input_filename)

    @pytest.fixture(autouse=True)
    def init_psf_file(self, tmpdir):
        self.psf_filename = str(tmpdir / 'psf.json')

        psf_pars = dict()
        psf_pars['psf1'] = dict(ampl=1., fwhm=7.0644601350928475)
        psf_pars['psf2'] = dict(ampl=0, fwhm=1E-5)
        psf_pars['psf3'] = dict(ampl=0, fwhm=1E-5)
        with open(self.psf_filename, 'w') as fh:
            json.dump(psf_pars, fh)

    @pytest.mark.parametrize('scale', SCALES)
    def test_command_line_gammapy_image_ts(self, tmpdir, scale):
        """Minimal test of gammapy_image_ts using testcase that
        guaranteed to work with compute_ts_image"""

        actual_filename = str(tmpdir / 'output.fits')

        args = [self.input_filename, actual_filename,
                "--psf", self.psf_filename,
                "--scales", scale]
        image_ts_main(args)

        expected_filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/expected_ts_{}.fits.gz'
        expected_filename = expected_filename.format(scale)

        actual = SkyImageCollection.read(actual_filename)
        expected = SkyImageCollection.read(expected_filename)

        opts = dict(rtol=1e-2, atol=1e-15, equal_nan=True)
        assert_allclose(actual['ts'].data, expected['ts'].data, **opts)
        assert_allclose(actual['sqrt_ts'].data, expected['sqrt_ts'].data, **opts)
        assert_allclose(actual['amplitude'].data, expected['amplitude'].data, **opts)
        assert_allclose(actual['niter'].data, expected['niter'].data, **opts)

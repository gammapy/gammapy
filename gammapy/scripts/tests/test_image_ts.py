# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import json
from numpy.testing.utils import assert_allclose
from astropy.tests.helper import pytest
from astropy.io import fits
from ...utils.testing import requires_dependency, requires_data
from ...image import SkyImageCollection
from ...datasets import load_poisson_stats_image
from ...datasets.core import gammapy_extra
from ..image_ts import image_ts_main

SCALES = ['0.000', '0.050', '0.100', '0.200']


@requires_dependency('scipy')
@requires_dependency('skimage')
@requires_data('gammapy-extra')
class TestImageTS:
    @pytest.fixture(autouse=True)
    def init_test_input_file(self, tmpdir):
        filename = str(tmpdir / 'input_all.fits.gz')

        data = load_poisson_stats_image(extra_info=True)
        header = data['header']
        images = [data['counts'], data['background'], data['exposure']]
        image_names = ['Counts', 'Background', 'Exposure']

        hdu_list = fits.HDUList()
        for image, name in zip(images, image_names):
            hdu = fits.ImageHDU(data=image, header=header, name=name)
            hdu_list.append(hdu)
        hdu_list.writeto(filename)
        self.input_file = filename

    @pytest.fixture(autouse=True)
    def init_psf_file(self, tmpdir):
        psf_filename = str(tmpdir / 'psf.json')

        psf_pars = dict()
        psf_pars['psf1'] = dict(ampl=1., fwhm=7.0644601350928475)
        psf_pars['psf2'] = dict(ampl=0, fwhm=1E-5)
        psf_pars['psf3'] = dict(ampl=0, fwhm=1E-5)
        json.dump(psf_pars, open(psf_filename, "w"))
        self.psf_file = psf_filename

    @pytest.mark.parametrize('scale', SCALES)
    def test_command_line_gammapy_image_ts(self, tmpdir, scale):
        """Minimal test of gammapy_image_ts using testcase that
        guaranteed to work with compute_ts_image"""

        output_filename = str(tmpdir / "output.fits")
        expected_filename = str(gammapy_extra.dir /
                                'test_datasets/unbundled/poisson_stats_image/expected_ts.fits.gz')

        image_ts_main([self.input_file,
                       output_filename,
                       "--psf",
                       self.psf_file,
                       "--scales",
                       scale])

        expected_filename_ = expected_filename.replace('.fits', '_{0}.fits'.format(scale))
        actual = SkyImageCollection.read(output_filename)
        expected = SkyImageCollection.read(expected_filename_)

        assert_allclose(actual['ts'].data, expected['ts'].data, rtol=1e-2, atol=1e-15,
                        equal_nan=True)
        assert_allclose(actual['sqrt_ts'].data, expected['sqrt_ts'].data,
                        rtol=1e-2, atol=1e-15, equal_nan=True)
        assert_allclose(actual['amplitude'].data, expected['amplitude'].data,
                        rtol=1e-2, atol=1e-15, equal_nan=True)
        assert_allclose(actual['niter'].data, expected['niter'].data,
                        rtol=1e-2, atol=1e-15, equal_nan=True)

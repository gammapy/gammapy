# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import os
from numpy.testing.utils import assert_allclose
from astropy.tests.helper import pytest
from astropy.io import fits
from ...datasets import load_poisson_stats_image
from ..sherpa_like import sherpa_image_like

try:
    import sherpa
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False


@pytest.mark.skipif('not HAS_SHERPA')
def test_sherpa_like(tmpdir):
    # load test data
    filenames = load_poisson_stats_image(extra_info=True, return_filenames=True)
    outfile = str(tmpdir.join('test_sherpa_like.json'))

    # rewrite files as .fits, because sherpa can't handle .fits.gz
    def rewrite(filename, tmpdir):
        hdu_list = fits.open(filename)
        _, filename = os.path.split(filename)
        new_filename = str(tmpdir.join(filename.replace('fits.gz', 'fits')))
        hdu_list.writeto(new_filename)
        return new_filename

    # write test source json file
    sources_data = {}
    sources_data['gaussian'] = {'ampl': 1E3,
                                'xpos': 99,
                                'ypos': 99,
                                'fwhm': 4 * 2.3548}

    sources = str(tmpdir.join('test_sherpa_like_sources.json'))
    with open(sources, 'w') as f:
        json.dump(sources_data, f)

    # set up args
    args = {'counts': rewrite(filenames['counts'], tmpdir),
            'exposure': rewrite(filenames['exposure'], tmpdir),
            'background': rewrite(filenames['background'], tmpdir),
            'psf': filenames['psf'],
            'sources': sources,
            'roi': None,
            'outfile': outfile}
    sherpa_image_like(**args)

    with open(outfile, 'r') as f:
        data = json.load(f)
        assert_allclose(data['fit']['parvals'], [9.016334, 99.365574,
                                                 99.647234, 10.97365])

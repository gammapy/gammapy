# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import json
from numpy.testing.utils import assert_allclose
import pytest

from astropy.stats import gaussian_sigma_to_fwhm
from ...utils.testing import requires_dependency, requires_data
from ...datasets import load_poisson_stats_image
from ..image_fit import run_image_fit_sherpa

EXPECTED = ([9.016526, 99.865985, 100.147877, 1010.824189],
            [4 * gaussian_sigma_to_fwhm, 100, 100, 1e3],
            [5 * gaussian_sigma_to_fwhm, 100, 100, 1e3])
RTOL = (1e-5, 1e-3, 1e-3)
PSF = (True, True, False)
DATA = (True, False, False)


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
@pytest.mark.parametrize('expected, rtol, psf, data',
                         zip(EXPECTED, RTOL, PSF, DATA))
def test_sherpa_like(tmpdir, expected, rtol, psf, data):
    """
    Fit Poisson stats image test data.
    """

    # load test data
    filenames = load_poisson_stats_image(extra_info=True, return_filenames=True)
    outfile = tmpdir / 'test_sherpa_like.json'

    # write test source json file
    sources_data = {}
    sources_data['gaussian'] = {'ampl': 1e3,
                                'xpos': 99,
                                'ypos': 99,
                                'fwhm': 4 * gaussian_sigma_to_fwhm}

    filename = tmpdir / 'test_sherpa_like_sources.json'
    with filename.open('w') as fh:
        json.dump(sources_data, fh)

    # set up args
    args = {'exposure': str(filenames['exposure']),
            'background': str(filenames['background']),
            'sources': str(filename),
            'roi': None,
            'outfile': str(outfile)}

    if data:
        args['counts'] = str(filenames['counts'])
    else:
        args['counts'] = str(filenames['model'])

    if psf:
        args['psf'] = filenames['psf']
    else:
        args['psf'] = None

    run_image_fit_sherpa(**args)

    with outfile.open() as fh:
        data = json.load(fh)

    # This recovers the values from the test dataset documented here:
    # https://github.com/gammapy/gammapy-extra/tree/master/
    # test_datasets/unbundled/poisson_stats_image#data
    actual = data['fit']['parvals']
    assert_allclose(actual, expected, rtol=rtol)

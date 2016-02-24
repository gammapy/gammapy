# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import Energy
from ...irf import PSFKing

test_psf_king_args = [
    {
        'chain': 'hap-hd',
        'filename': '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_psf_king_023523.fits.gz',
        'energy': 0.25178512930870056,
        'offset': 1.25,
        'gamma': 3.8902931213378906,
        'sigma': 0.04095194861292839,
    },
    {
        'chain': 'pa',
        'filename': '$GAMMAPY_EXTRA/datasets/hess-crab4-pa/run23400-23599/run23523/psf_king_23523.fits.gz',
        'energy': 1.2574334144592285,
        'offset': 1.0,
        'gamma': 2.02886700630188,
        'sigma': 0.021144593134522438,
    }
]


@pytest.mark.parametrize('args', test_psf_king_args)
@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_psf_king(args):
    psf = PSFKing.read(args['filename'])
    psf.info()

    assert_quantity_allclose(psf.offset[2], Angle(args['offset'], 'deg'))
    assert_quantity_allclose(psf.energy[5], Energy(args['energy'], 'TeV'))
    assert_quantity_allclose(psf.gamma[2, 5], args['gamma'])
    assert_quantity_allclose(psf.sigma[2, 5], Angle(args['sigma'], 'deg'))

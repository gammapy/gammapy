# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from ...irf import PSFKing
from ...utils.testing import requires_dependency, requires_data


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_psf_king(args):
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_psf_king_023523.fits.gz',
    psf = PSFKing.read(filename)
    psf.info()

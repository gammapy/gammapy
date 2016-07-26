# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...utils.testing import requires_data, requires_dependency
from ...scripts import CTAIrf


@requires_data('gammapy-extra')
def test():
    irf_path = os.getenv('GAMMAPY_EXTRA') + \
        '/datasets/cta/perf_prod2/South_5h/irf_file.fits.gz'

    irf = CTAIrf.read(irf_path)

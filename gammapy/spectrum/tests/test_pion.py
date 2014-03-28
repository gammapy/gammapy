# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ..pion import PionDecaySpectrum

# TODO: implement
@pytest.mark.xfail
def test_PionDecaySpectrum():
    spectrum = PionDecaySpectrum(42)
    spectrum([42])
    spectrum([42, 43])

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.tests.helper import pytest
from ..core import GammaSpectralCube

try:
    import spectral_cube
    HAS_SPECTRAL_CUBE = True
except ImportError:
    HAS_SPECTRAL_CUBE = False


@pytest.mark.skipif('not HAS_SPECTRAL_CUBE')
def test_GammaSpectralCube():
    #spectral_cube = GammaSpectralCube()
    pass

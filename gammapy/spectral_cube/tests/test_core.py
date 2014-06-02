# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.tests.helper import pytest, remote_data
from ..core import GammaSpectralCube
from ...datasets import get_fermi_diffuse_background_model

'''
try:
    import spectral_cube
    HAS_SPECTRAL_CUBE = True
except ImportError:
    HAS_SPECTRAL_CUBE = False
@pytest.mark.skipif('not HAS_SPECTRAL_CUBE')
'''

@remote_data
def test_GammaSpectralCube():
    filename = get_fermi_diffuse_background_model()
    spectral_cube = GammaSpectralCube(filename)
    assert spectral_cube.data.shape == (30, 360, 720)

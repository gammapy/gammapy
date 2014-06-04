# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.tests.helper import pytest
from astropy.units import Quantity
from ..core import GammaSpectralCube
from ...datasets import FermiGalacticCenter
from ...utils.testing import assert_quantity


try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestGammaSpectralCube():
    
    def setup(self):
        self.spectral_cube = FermiGalacticCenter.diffuse_model()
        assert self.spectral_cube.data.shape == (30, 21, 61)

    def test_init(self):
        data = self.spectral_cube.data
        wcs = self.spectral_cube.wcs
        energy = self.spectral_cube.energy

        spectral_cube = GammaSpectralCube(data, wcs, energy)
        assert spectral_cube.data.shape == (30, 21, 61)

    def test_flux(self):
        lon = Quantity(0, 'deg')
        lat = Quantity(0, 'deg')
        energy = Quantity(1, 'GeV')
        
        flux = self.spectral_cube.flux(lon, lat, energy)
        assert_quantity(flux, Quantity(2.48101719e-05, '1 / (cm2 MeV s sr)'))

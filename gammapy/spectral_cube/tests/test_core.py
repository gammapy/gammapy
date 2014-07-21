# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.units import Quantity
from ..core import GammaSpectralCube
from ...datasets import FermiGalacticCenter
from ...utils.testing import assert_quantity


try:
    # The scipy.interpolation.RegularGridInterpolator class was added in Scipy version 0.14
    from scipy.interpolate import RegularGridInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    # The scipy.interpolation.RegularGridInterpolator class was added in Scipy version 0.14
    from FITS_tools.cube_regrid import regrid_cube_hdu
    HAS_FITSTOOLS = True
except ImportError:
    HAS_FITSTOOLS = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_correlated_fermi_psf():
    pass


def test_interp_flux():
    pass


def test_interp_exposure():
    pass


@pytest.mark.skipif('not HAS_FITSTOOLS')
def test_reproject_cube():
    pass


@pytest.mark.skipif('not HAS_FITSTOOLS')
def test_compute_npred_cube():
    pass


@pytest.mark.skipif('not HAS_SCIPY')
class TestGammaSpectralCube(object):

    def setup(self):
        self.spectral_cube = FermiGalacticCenter.diffuse_model()
        assert self.spectral_cube.data.shape == (30, 21, 61)

    def test_init(self):
        data = self.spectral_cube.data
        wcs = self.spectral_cube.wcs
        energy = self.spectral_cube.energy

        spectral_cube = GammaSpectralCube(data, wcs, energy)
        assert spectral_cube.data.shape == (30, 21, 61)

    def test_pix2world(self):
        # Corner pixel with index [0, 0, 0]
        lon, lat, energy = self.spectral_cube.pix2world(0, 0, 0)
        assert_quantity(lon, Quantity(344.75, 'deg'))
        assert_quantity(lat, Quantity(-5.25, 'deg'))
        assert_quantity(energy, Quantity(50, 'MeV'))

    def test_world2pix(self):
        lon = Quantity(344.75, 'deg')
        lat = Quantity(-5.25, 'deg')
        energy = Quantity(50, 'MeV')
        x, y, z = self.spectral_cube.world2pix(lon, lat, energy)
        assert_allclose((x, y, z), (0, 0, 0))

    def test_pix2world2pix(self):
        # Test round-tripping
        pix = 2.2, 3.3, 4.4
        world = self.spectral_cube.pix2world(*pix)
        pix2 = self.spectral_cube.world2pix(*world)
        assert_allclose(pix2, pix)

        # Check array inputs
        pix = [2.2, 2.2], [3.3, 3.3], [4.4, 4.4]
        world = self.spectral_cube.pix2world(*pix)
        pix2 = self.spectral_cube.world2pix(*world)
        assert_allclose(pix2, pix)

    @pytest.mark.xfail
    def test_flux_scalar(self):
        # Corner pixel with index [0, 0, 0]
        lon = Quantity(344.75, 'deg')  # pixel 0
        lat = Quantity(-5.25, 'deg')  # pixel 0
        energy = Quantity(50, 'MeV')  # slice 0
        actual = self.spectral_cube.flux(lon, lat, energy)
        expected = self.spectral_cube.data[0, 0, 0]
        assert_quantity(actual, expected)

        # Galactic center position
        lon = Quantity(0, 'deg')  # beween pixel 11 and 12 in ds9 viewer
        lat = Quantity(0, 'deg')  # beween pixel 30 and 31 in ds9 viewer
        energy = Quantity(528.9657943133443, 'MeV')  # slice 10 in ds9 viewer
        actual = self.spectral_cube.flux(lon, lat, energy)
        # Compute expected value by interpolating 4 neighbors
        # Use data axis order: energy, lat, lon
        # and remember that numpy starts counting at 0 whereas FITS start at 1
        s = self.spectral_cube.data
        expected = s[9, 10:12, 29:31].mean()

        # TODO: why are these currently inconsistent by a few % !?
        # actual   =  9.67254380e-07
        # expected = 10.13733026e-07
        assert_quantity(actual, expected)

    def test_flux_mixed(self):
        # Corner pixel with index [0, 0, 0]
        lon = Quantity([344.75, 344.75], 'deg')  # pixel 0 twice
        lat = Quantity([-5.25, -5.25], 'deg')  # pixel 0 twice
        energy = Quantity(50, 'MeV')  # slice 0
        actual = self.spectral_cube.flux(lon, lat, energy)
        expected = self.spectral_cube.data[0, 0, 0]
        assert_quantity(actual, expected)

    def test_flux_array(self):
        pix = [2, 2], [3, 3], [4, 4]
        world = self.spectral_cube.pix2world(*pix)
        actual = self.spectral_cube.flux(*world)
        expected = self.spectral_cube.data[4, 3, 2]
        # Quantity([3.50571123e-07, 2], '1 / (cm2 MeV s sr)')
        assert_quantity(actual, expected)

    def test_integral_flux_image(self):
        # For a very small energy band the integral flux should be roughly
        # differential flux times energy bin width
        lon, lat, energy = self.spectral_cube.pix2world(0, 0, 0)
        denergy = 0.001 * energy
        energy_band = Quantity([energy, energy + denergy])
        dflux = self.spectral_cube.flux(lon, lat, energy)
        expected = (dflux * denergy).to('cm^-2 s^-1 sr^-1').value
        actual = self.spectral_cube.integral_flux_image(energy_band).data[0, 0]
        assert_allclose(actual, expected, rtol=1e-3)

        # Test a wide energy band
        energy_band = Quantity([1, 10], 'GeV')
        image = self.spectral_cube.integral_flux_image(energy_band)
        actual = image.data.sum()
        # TODO: the reference result is not verified ... just pasted from the test output.
        expected = 5.2481972772213124e-05
        assert_allclose(actual, expected)

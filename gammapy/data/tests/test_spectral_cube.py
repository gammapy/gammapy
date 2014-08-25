# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from astropy.tests.helper import pytest
from astropy.units import Quantity
from astropy.wcs import WCS
from ...datasets import FermiGalacticCenter, FermiVelaRegion
from ...data import SpectralCube, compute_npred_cube, convolve_cube
from ...image import solid_angle, make_header, make_empty_image
from ...irf import EnergyDependentTablePSF
from ...spectrum.powerlaw import power_law_eval
from ...utils.testing import assert_quantity


try:
    # The scipy.interpolation.RegularGridInterpolator class was added in Scipy version 0.14
    from scipy.interpolate import RegularGridInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from reproject import reproject
    HAS_REPROJECT = True
except ImportError:
    HAS_REPROJECT = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestSpectralCube(object):

    def setup(self):
        self.spectral_cube = FermiGalacticCenter.diffuse_model()
        assert self.spectral_cube.data.shape == (30, 21, 61)

    def test_init(self):
        data = self.spectral_cube.data
        wcs = self.spectral_cube.wcs
        energy = self.spectral_cube.energy

        spectral_cube = SpectralCube(data, wcs, energy)
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
        expected = (dflux * denergy).to('cm^-2 s^-1 sr^-1')
        actual = self.spectral_cube.integral_flux_image(energy_band).data[0, 0]
        assert_quantity(actual, expected, rtol=1e-3)

        # Test a wide energy band
        energy_band = Quantity([1, 10], 'GeV')
        image = self.spectral_cube.integral_flux_image(energy_band)
        actual = image.data.sum()
        # TODO: the reference result is not verified ... just pasted from the test output.
        expected = 5.2481972772213124e-05
        assert_allclose(actual, expected)

    def test_solid_angle_image(self):
        actual = self.spectral_cube.solid_angle_image[10][30]
        expected = Quantity(0.24999762018018362, 'steradian')
        assert_quantity(actual, expected)

    def test_spatial_coordinate_images(self):
        lon, lat = self.spectral_cube.spatial_coordinate_images

        assert lon.shape == (21, 61)
        assert lat.shape == (21, 61)

        # TODO assert the four corner values


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_REPROJECT')
def test_compute_npred_cube():
    # A quickly implemented check - should be improved
    filenames = FermiGalacticCenter.filenames()
    spectral_cube = SpectralCube.read(filenames['diffuse_model'])
    exposure_cube = SpectralCube.read(filenames['exposure_cube'])
    counts_cube = FermiGalacticCenter.counts()
    energy_bounds = Quantity([10, 30, 100, 500], 'GeV')

    spectral_cube = spectral_cube.reproject_to(exposure_cube)

    npred_cube = compute_npred_cube(spectral_cube,
                                    exposure_cube,
                                    energy_bounds)
    expected_sum = counts_cube.data.sum()
    actual_sum = np.nan_to_num(npred_cube.data).sum()
    # Check npredicted is same order of magnitude of true counts
    assert_allclose(expected_sum, actual_sum, rtol=1)
    # PSF convolve the npred cube
    psf = EnergyDependentTablePSF.read(FermiGalacticCenter.filenames()['psf'])
    npred_cube_convolved = convolve_cube(npred_cube, psf, offset_max=Angle(3, 'deg'))
    actual_convolved_sum = npred_cube_convolved.data.sum()
    # Check sum is the same after convolution
    assert_allclose(actual_sum, actual_convolved_sum, rtol=0.1)
    # Test shape
    expected = ((len(energy_bounds) - 1, exposure_cube.data.shape[1],
                 exposure_cube.data.shape[2]))
    actual = npred_cube_convolved.data.shape
    assert_allclose(actual, expected)


def make_test_cubes(energies, nxpix, nypix, binsz):
    """Makes exposure and spectral cube for tests.
    Parameters
    ----------
    energies : `~astropy.units.Quantity`
        Quantity 1D array of energies of cube layers
    nxpix : int
        Number of pixels in x-spatial direction
    nypix : int
        Number of pixels in y-spatial direction
    binsz : float
        Spatial resolution of cube, in degrees per pixel

    Returns
    -------
    exposure_cube : `~gammapy.spectral_cube.SpectralCube`
        Cube of uniform exposure = 1 cm^2 s
    spectral_cube : `~gammapy.spectral_cube.SpectralCube`
        Cube of differential fluxes in units of cm^-2 s^-1 GeV^-1 sr^-1
    """
    hdu = make_empty_image(nxpix, nypix, binsz)
    solid_angle_array = solid_angle(hdu)
    header = make_header(nxpix, nypix, binsz)
    header['NAXIS'] = 3
    header['NAXIS3'] = len(energies)
    header['CDELT3'] = 1
    header['CRVAL3'] = 1
    header['CRPIX3'] = 1
    wcs = WCS(header)
    data_array = np.ones((len(energies), 10, 10))
    exposure_cube = SpectralCube(data=Quantity(data_array, 'cm2 s'),
                                 wcs=wcs, energy=energies)

    flux = Quantity(power_law_eval(energies.value, 1, 2,
                                   1), '1/(cm2 s GeV sr)')
    flux_array = np.zeros_like(data_array)
    for i in np.arange(len(flux)):
        flux_array[i] = flux.value[i] * data_array[i]
    spectral_cube = SpectralCube(data=Quantity(flux_array, flux.unit),
                                 wcs=wcs, energy=energies)
    return exposure_cube, spectral_cube


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_REPROJECT')
def test_analytical_npred_cube():
    # Analytical check: g=2, N=1 gives int. flux 0.25 between 1 and 2
    # (arbitrary units of energy).
    # Exposure = 1, so solid angle only factor which varies.
    # Result should be 0.5 * 1 * solid_angle_array from integrating analytically

    energies = Quantity([1, 2], 'GeV')
    exposure_cube, spectral_cube = make_test_cubes(energies, 10, 10, 1)

    solid_angle_array = exposure_cube.solid_angle_image
    # Expected npred counts (so no quantity)
    expected = 0.5 * solid_angle_array.value
    # Integral resolution is 1 as this is a true powerlaw case
    npred_cube = compute_npred_cube(spectral_cube, exposure_cube,
                                    energies, integral_resolution=1)

    actual = npred_cube.data[0]

    assert_allclose(actual, expected)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_REPROJECT')
def test_convolve_cube():
    filenames = FermiGalacticCenter.filenames()
    spectral_cube = SpectralCube.read(filenames['diffuse_model'])
    exposure_cube = SpectralCube.read(filenames['exposure_cube'])
    energy_bounds = Quantity([10, 30, 100, 500], 'GeV')

    spectral_cube = spectral_cube.reproject_to(exposure_cube)

    npred_cube = compute_npred_cube(spectral_cube,
                                    exposure_cube,
                                    energy_bounds)
    # PSF convolve the npred cube
    psf = EnergyDependentTablePSF.read(FermiGalacticCenter.filenames()['psf'])
    npred_cube_convolved = convolve_cube(npred_cube, psf, offset_max=Angle(5, 'deg'))
    expected = npred_cube.data.sum()
    actual = npred_cube_convolved.data.sum()

    assert_allclose(actual, expected, rtol=1e-2)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_REPROJECT')
def test_reproject_cube():
    # TODO: a better test can probably be implemented here to avoid
    # repeating code
    filenames = FermiGalacticCenter.filenames()
    spectral_cube = SpectralCube.read(filenames['diffuse_model'])
    exposure_cube = SpectralCube.read(filenames['exposure_cube'])

    original_cube = Quantity(np.nan_to_num(spectral_cube.data.value),
                             spectral_cube.data.unit)
    spectral_cube = spectral_cube.reproject_to(exposure_cube)
    reprojected_cube = Quantity(np.nan_to_num(spectral_cube.data.value),
                                spectral_cube.data.unit)
    # 0.5 degrees per pixel in diffuse model
    # 2 degrees in reprojection reference
    # sum of reprojected should be 1/16 of sum of original if flux-preserving
    expected = 0.0625 * original_cube.sum()
    actual = reprojected_cube.sum()

    assert_quantity(actual, expected, rtol=1e-2)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.units import Quantity
from astropy.wcs import WCS
from ...utils.testing import requires_dependency, requires_data
from ...datasets import FermiGalacticCenter
from ...image import make_header
from ...irf import EnergyDependentTablePSF
from ...spectrum.powerlaw import power_law_evaluate
from .. import SkyCube, compute_npred_cube, convolve_cube


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestSkyCube(object):
    def setup(self):
        self.sky_cube = FermiGalacticCenter.diffuse_model()
        assert self.sky_cube.data.shape == (30, 21, 61)

    def test_init(self):
        name = 'Axel'
        data = self.sky_cube.data
        wcs = self.sky_cube.wcs
        energy = self.sky_cube.energy

        sky_cube = SkyCube(name, data, wcs, energy)
        assert sky_cube.data.shape == (30, 21, 61)

    def test_read_write(self):
        filename = 'sky_cube_test.fits'
        self.sky_cube.writeto(filename)

        sky_cube = SkyCube.read(filename)
        assert sky_cube.data.shape == (30, 21, 61)

    def test_pix2world(self):
        # Corner pixel with index [0, 0, 0]
        lon, lat, energy = self.sky_cube.pix2world(0, 0, 0)
        assert_quantity_allclose(lon, Quantity(344.75, 'deg'))
        assert_quantity_allclose(lat, Quantity(-5.25, 'deg'))
        assert_quantity_allclose(energy, Quantity(50, 'MeV'))

    def test_world2pix(self):
        lon = Quantity(344.75, 'deg')
        lat = Quantity(-5.25, 'deg')
        energy = Quantity(50, 'MeV')
        x, y, z = self.sky_cube.world2pix(lon, lat, energy)
        assert_allclose((x, y, z), (0, 0, 0))

    def test_pix2world2pix(self):
        # Test round-tripping
        pix = 2.2, 3.3, 4.4
        world = self.sky_cube.pix2world(*pix)
        pix2 = self.sky_cube.world2pix(*world)
        assert_allclose(pix2, pix)

        # Check array inputs
        pix = [2.2, 2.2], [3.3, 3.3], [4.4, 4.4]
        world = self.sky_cube.pix2world(*pix)
        pix2 = self.sky_cube.world2pix(*world)
        assert_allclose(pix2, pix)

    @pytest.mark.xfail
    def test_flux_scalar(self):
        # Corner pixel with index [0, 0, 0]
        lon = Quantity(344.75, 'deg')  # pixel 0
        lat = Quantity(-5.25, 'deg')  # pixel 0
        energy = Quantity(50, 'MeV')  # slice 0
        actual = self.sky_cube.flux(lon, lat, energy)
        expected = self.sky_cube.data[0, 0, 0]
        assert_quantity_allclose(actual, expected)

        # Galactic center position
        lon = Quantity(0, 'deg')  # beween pixel 11 and 12 in ds9 viewer
        lat = Quantity(0, 'deg')  # beween pixel 30 and 31 in ds9 viewer
        energy = Quantity(528.9657943133443, 'MeV')  # slice 10 in ds9 viewer
        actual = self.sky_cube.flux(lon, lat, energy)
        # Compute expected value by interpolating 4 neighbors
        # Use data axis order: energy, lat, lon
        # and remember that numpy starts counting at 0 whereas FITS start at 1
        s = self.sky_cube.data
        expected = s[9, 10:12, 29:31].mean()

        # TODO: why are these currently inconsistent by a few % !?
        # actual   =  9.67254380e-07
        # expected = 10.13733026e-07
        assert_quantity_allclose(actual, expected)

    def test_flux_mixed(self):
        # Corner pixel with index [0, 0, 0]
        lon = Quantity([344.75, 344.75], 'deg')  # pixel 0 twice
        lat = Quantity([-5.25, -5.25], 'deg')  # pixel 0 twice
        energy = Quantity(50, 'MeV')  # slice 0
        actual = self.sky_cube.flux(lon, lat, energy)
        expected = self.sky_cube.data[0, 0, 0]
        assert_quantity_allclose(actual, expected)

    def test_flux_array(self):
        pix = [2, 2], [3, 3], [4, 4]
        world = self.sky_cube.pix2world(*pix)
        actual = self.sky_cube.flux(*world)
        expected = self.sky_cube.data[4, 3, 2]
        # Quantity([3.50571123e-07, 2], '1 / (cm2 MeV s sr)')
        assert_quantity_allclose(actual, expected)

    def test_integral_flux_image(self):
        # For a very small energy band the integral flux should be roughly
        # differential flux times energy bin width
        lon, lat, energy = self.sky_cube.pix2world(0, 0, 0)
        denergy = 0.001 * energy
        energy_band = Quantity([energy, energy + denergy])
        dflux = self.sky_cube.flux(lon, lat, energy)
        expected = dflux * denergy
        actual = Quantity(self.sky_cube.integral_flux_image(energy_band).data[0, 0],
                          '1 / (cm2 s sr)')
        assert_quantity_allclose(actual, expected, rtol=1e-3)

        # Test a wide energy band
        energy_band = Quantity([1, 10], 'GeV')
        image = self.sky_cube.integral_flux_image(energy_band)
        actual = image.data.sum()
        # TODO: the reference result is not verified ... just pasted from the test output.
        expected = 5.2481972772213124e-02
        assert_allclose(actual, expected)

        # Test integral flux for energy bands with units.
        energy_band_check = Quantity([1000, 10000], 'MeV')
        new_image = self.sky_cube.integral_flux_image(energy_band_check)
        assert_allclose(new_image.data, image.data)

        # Test Header Keys
        expected = [('CDELT1', 0.5), ('CDELT2', 0.5), ('NAXIS1', 61),
                    ('NAXIS2', 21), ('CRVAL1', 0), ('CRVAL2', 0)]

        for key, value in expected:
            assert_allclose(np.abs(image.header[key]), value)

    # TODO: fix this test.
    # It's currently failing. Dont' know which number (if any) is correct.
    # E        x: array(7.615363001210512e-05)
    # E        y: array(0.00015230870989335428)
    @pytest.mark.xfail
    def test_solid_angle_image(self):
        actual = self.sky_cube.solid_angle_image[10][30]
        expected = Quantity(self.sky_cube.wcs.wcs.cdelt[:-1].prod(), 'deg2')
        assert_quantity_allclose(actual, expected, rtol=1e-4)

    def test_spatial_coordinate_images(self):
        coordinates = self.sky_cube.spatial_coordinate_images()
        lon = coordinates.data.lon
        lat = coordinates.data.lat

        assert lon.shape == (21, 61)
        assert lat.shape == (21, 61)

        assert_allclose(lon[0, 0], Angle("344d45m00s"))
        assert_allclose(lat[0, 0], Angle(" -5d15m00s"))

        assert_allclose(lon[0, -1], Angle("14d45m00s"))
        assert_allclose(lat[0, -1], Angle("-5d15m00s"))

        assert_allclose(lon[-1, 0], Angle("344d45m00s"))
        assert_allclose(lat[-1, 0], Angle("4d45m00s"))

        assert_allclose(lon[-1, -1], Angle("14d45m00s"))
        assert_allclose(lat[-1, -1], Angle("4d45m00s"))


@pytest.mark.xfail
@requires_dependency('scipy.interpolate.RegularGridInterpolator')
@requires_dependency('reproject')
def test_compute_npred_cube():
    # A quickly implemented check - should be improved
    filenames = FermiGalacticCenter.filenames()
    sky_cube = SkyCube.read(filenames['diffuse_model'])
    exposure_cube = SkyCube.read(filenames['exposure_cube'])
    counts_cube = FermiGalacticCenter.counts()
    energy_bounds = Quantity([10, 30, 100, 500], 'GeV')

    sky_cube = sky_cube.reproject_to(exposure_cube)

    npred_cube = compute_npred_cube(sky_cube,
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
    exposure_cube : `~gammapy.sky_cube.SkyCube`
        Cube of uniform exposure = 1 cm^2 s
    sky_cube : `~gammapy.sky_cube.SkyCube`
        Cube of differential fluxes in units of cm^-2 s^-1 GeV^-1 sr^-1
    """
    header = make_header(nxpix, nypix, binsz)
    header['NAXIS'] = 3
    header['NAXIS3'] = len(energies)
    header['CDELT3'] = 1
    header['CRVAL3'] = 1
    header['CRPIX3'] = 1
    wcs = WCS(header)
    data_array = np.ones((len(energies), 10, 10))
    exposure_cube = SkyCube(data=Quantity(data_array, 'cm2 s'),
                            wcs=wcs, energy=energies)

    flux = power_law_evaluate(energies.value, 1, 2, 1)
    flux = Quantity(flux, '1/(cm2 s GeV sr)')
    flux_array = np.zeros_like(data_array)
    for i in np.arange(len(flux)):
        flux_array[i] = flux.value[i] * data_array[i]
    sky_cube = SkyCube(data=Quantity(flux_array, flux.unit),
                       wcs=wcs, energy=energies)
    return exposure_cube, sky_cube


@requires_dependency('scipy.interpolate.RegularGridInterpolator')
@requires_dependency('reproject')
def test_analytical_npred_cube():
    # Analytical check: g=2, N=1 gives int. flux 0.25 between 1 and 2
    # (arbitrary units of energy).
    # Exposure = 1, so solid angle only factor which varies.
    # Result should be 0.5 * 1 * solid_angle_array from integrating analytically

    energies = Quantity([1, 2], 'MeV')
    exposure_cube, sky_cube = make_test_cubes(energies, 10, 10, 1)

    solid_angle_array = exposure_cube.solid_angle_image
    # Expected npred counts (so no quantity)
    expected = 0.5 * solid_angle_array.value
    # Integral resolution is 1 as this is a true powerlaw case
    npred_cube = compute_npred_cube(sky_cube, exposure_cube,
                                    energies, integral_resolution=1)

    actual = npred_cube.data[0]

    assert_allclose(actual, expected)


@requires_dependency('scipy.interpolate.RegularGridInterpolator')
@requires_dependency('reproject')
def test_convolve_cube():
    filenames = FermiGalacticCenter.filenames()
    sky_cube = SkyCube.read(filenames['diffuse_model'])
    exposure_cube = SkyCube.read(filenames['exposure_cube'])
    energy_bounds = Quantity([10, 30, 100, 500], 'GeV')

    sky_cube = sky_cube.reproject_to(exposure_cube)

    npred_cube = compute_npred_cube(sky_cube,
                                    exposure_cube,
                                    energy_bounds)
    # PSF convolve the npred cube
    psf = EnergyDependentTablePSF.read(FermiGalacticCenter.filenames()['psf'])
    npred_cube_convolved = convolve_cube(npred_cube, psf, offset_max=Angle(5, 'deg'))
    expected = npred_cube.data.sum()
    actual = npred_cube_convolved.data.sum()

    assert_allclose(actual, expected, rtol=1e-2)


@pytest.mark.xfail
@requires_dependency('scipy')
@requires_dependency('reproject')
def test_reproject_cube():
    # TODO: a better test can probably be implemented here to avoid
    # repeating code
    filenames = FermiGalacticCenter.filenames()
    sky_cube = SkyCube.read(filenames['diffuse_model'])
    exposure_cube = SkyCube.read(filenames['exposure_cube'])

    original_cube = Quantity(np.nan_to_num(sky_cube.data.value),
                             sky_cube.data.unit)
    sky_cube = sky_cube.reproject_to(exposure_cube)
    reprojected_cube = Quantity(np.nan_to_num(sky_cube.data.value),
                                sky_cube.data.unit)
    # 0.5 degrees per pixel in diffuse model
    # 2 degrees in reprojection reference
    # sum of reprojected should be 1/16 of sum of original if flux-preserving
    expected = 0.0625 * original_cube.sum()
    actual = reprojected_cube.sum()

    assert_quantity_allclose(actual, expected, rtol=1e-2)

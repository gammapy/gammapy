# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import textwrap

import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from astropy.tests.helper import pytest, assert_quantity_allclose
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from ...utils.testing import requires_dependency, requires_data
from ...data import EventList
from ...datasets import FermiGalacticCenter
from ...image import make_header
from ...irf import EnergyDependentTablePSF
from ...spectrum.powerlaw import power_law_evaluate
from ...spectrum.models import PowerLaw2
from .. import SkyCube, compute_npred_cube, convolve_cube


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestSkyCube(object):
    def setup(self):
        self.sky_cube = FermiGalacticCenter.diffuse_model()
        assert self.sky_cube.data.shape == (30, 21, 61)

    def test_to_images(self):
        images = self.sky_cube.to_images()
        cube = images.to_cube()
        SkyCube.assert_allclose(self.sky_cube, cube)

    def test_init(self):
        name = 'Axel'
        data = self.sky_cube.data
        wcs = self.sky_cube.wcs
        energy = self.sky_cube.energy_axis.energy

        sky_cube = SkyCube(name, data, wcs, energy)
        assert sky_cube.data.shape == (30, 21, 61)

    def test_read_write(self, tmpdir):
        filename = str(tmpdir / 'sky_cube.fits')
        self.sky_cube.write(filename, format='fermi-background')

        sky_cube = SkyCube.read(filename, format='fermi-background')
        assert sky_cube.data.shape == (30, 21, 61)

    def test_pixel_to_skycoord(self):
        # Corner pixel with index [0, 0, 0]
        position, energy = self.sky_cube.wcs_pixel_to_skycoord(0, 0, 0)
        lon, lat = position.galactic.l, position.galactic.b
        assert_quantity_allclose(lon, 344.75 * u.deg)
        assert_quantity_allclose(lat, -5.25 * u.deg)
        assert_quantity_allclose(energy, 50 * u.MeV)

    def test_skycoord_to_pixel(self):
        position = SkyCoord(344.75, -5.25, frame='galactic', unit='deg')
        energy = 50 * u.MeV
        x, y, z = self.sky_cube.wcs_skycoord_to_pixel(position, energy)
        assert_allclose((x, y, z), (0, 0, 0))

    def test_pix2world2pix(self):
        # Test round-tripping
        pix = 2.2, 3.3, 4.4
        world = self.sky_cube.wcs_pixel_to_skycoord(*pix)
        pix2 = self.sky_cube.wcs_skycoord_to_pixel(*world)
        assert_allclose(pix2, pix)

        # Check array inputs
        pix = [2.2, 2.2], [3.3, 3.3], [4.4, 4.4]
        world = self.sky_cube.wcs_pixel_to_skycoord(*pix)
        pix2 = self.sky_cube.wcs_skycoord_to_pixel(*world)
        assert_allclose(pix2, pix)

    def test_lookup(self):
        # Corner pixel with index [0, 0, 0]
        position = SkyCoord(344.75, -5.25, frame='galactic', unit='deg')
        energy = 50 * u.MeV  # slice 0
        actual = self.sky_cube.lookup(position, energy)
        expected = self.sky_cube.data[0, 0, 0]
        assert_quantity_allclose(actual, expected)

    def test_lookup_array(self):
        pix = [2, 2], [3, 3], [4, 4]
        position, energy = self.sky_cube.wcs_pixel_to_skycoord(*pix)
        actual = self.sky_cube.lookup(position, energy)
        expected = self.sky_cube.data[4, 3, 2]
        # Quantity([3.50571123e-07, 2], '1 / (cm2 MeV s sr)')
        assert_quantity_allclose(actual, expected)

    def test_sky_image_integral(self):
        # For a very small energy band the integral flux should be roughly
        # differential flux times energy bin width
        position, energy = self.sky_cube.wcs_pixel_to_skycoord(0, 0, 0)
        denergy = 0.001 * energy
        emin, emax = energy, energy + denergy
        dflux = self.sky_cube.lookup(position, energy, interpolation='linear')
        expected = dflux * denergy
        actual = self.sky_cube.sky_image_integral(emin, emax, nbins=100)
        assert_quantity_allclose(actual.data[0, 0], expected, rtol=1e-3)

        # Test a wide energy band
        emin, emax = [1, 10] * u.GeV
        image = self.sky_cube.sky_image_integral(emin, emax, nbins=100)
        unit = '1 / (s sr cm2)'
        actual = image.data.sum().to(unit)
        # TODO: the reference result is not verified ... just pasted from the test output.
        expected = 0.05098313774120132 * u.Unit(unit)
        assert_allclose(actual, expected)

    def test_repr(self):
        actual = repr(self.sky_cube)
        expected = textwrap.dedent("""\
        Sky cube flux with shape=(30, 21, 61) and unit=1 / (cm2 MeV s sr):
         n_lon:       61  type_lon:    GLON-CAR         unit_lon:    deg
         n_lat:       21  type_lat:    GLAT-CAR         unit_lat:    deg
         n_energy:    30  unit_energy: MeV
        """)
        assert actual == expected

@requires_dependency('scipy')
class TestSkyCubeInterpolation(object):
    def setup(self):
        # Set up powerlaw
        amplitude = 1E-12 * u.Unit('1 / (s sr cm2)')
        index = 2
        emin = 1 * u.TeV
        emax = 100 * u.TeV
        pwl = PowerLaw2(amplitude, index, emin, emax)

        # Set up data cube
        cube = SkyCube.empty(emin=emin, emax=emax, enumbins=4, nxpix=3, nypix=3)
        data = pwl(cube.energies()).reshape(-1, 1, 1) * np.ones(cube.data.shape[1:])
        cube.data = data
        self.sky_cube = cube
        self.pwl = pwl

    def test_sky_image(self):
        energy = 50 * u.TeV
        image = self.sky_cube.sky_image(energy, interpolation='linear')
        assert_quantity_allclose(image.data, self.pwl(energy))

    def test_sky_image_integrate(self):
        emin, emax = [1, 100] * u.TeV
        integral = self.sky_cube.sky_image_integral(emin, emax)
        assert_quantity_allclose(integral.data, self.pwl.integral(emin, emax))

    @requires_dependency('reproject')
    def test_reproject(self):
        emin = 1 * u.TeV
        emax = 100 * u.TeV
        ref = SkyCube.empty(emin=emin, emax=emax, enumbins=4, nxpix=6, nypix=6,
                            binsz=0.01)
        reprojected = self.sky_cube.reproject(ref)

        # Check if reprojection conserves total flux
        integral = self.sky_cube.sky_image_integral(emin, emax)
        flux = (integral.data * integral.solid_angle()).sum()

        integral_rep = reprojected.sky_image_integral(emin, emax)
        flux_rep = (integral_rep.data * integral_rep.solid_angle()).sum()

        assert_quantity_allclose(flux, flux_rep)



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

    solid_angle_array = exposure_cube.solid_angle
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


@requires_data('gammapy-extra')
def test_bin_events_in_cube():
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_events_023523.fits.gz'
    events = EventList.read(filename)
    counts = SkyCube.empty(emin=0.5, emax=80, enumbins=8, eunit='TeV',
                           nxpix=200, nypix=200, xref=events.meta['RA_OBJ'],
                           yref=events.meta['DEC_OBJ'], dtype='int',
                           coordsys='CEL')

    counts.fill_events(events)

    assert counts.data.sum() == 1233

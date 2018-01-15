# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import textwrap
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import Energy, EnergyBounds
from ...image import SkyImage
from ...data import EventList
from ...datasets import FermiGalacticCenter
from ...spectrum.models import PowerLaw2
from .. import SkyCube


def make_test_spectral_model():
    emin, emax = 1 * u.TeV, 100 * u.TeV,
    return PowerLaw2(
        amplitude=1e-12 * u.Unit('1 / (s sr cm2)'),
        index=2,
        emin=emin,
        emax=emax,
    )


def make_test_sky_cube():
    model = make_test_spectral_model()
    emin = model.parameters['emin'].value
    emax = model.parameters['emax'].value
    cube = SkyCube.empty(emin=emin, emax=emax, enumbins=4, nxpix=3, nypix=3)
    data = model(cube.energies()).reshape(-1, 1, 1) * np.ones(cube.data.shape[1:])
    cube.data = data
    return cube


@requires_dependency('scipy')
def test_empty_like_energy():
    image = SkyImage.empty(nxpix=11, nypix=7)
    energies = Energy.equal_log_spacing(1 * u.TeV, 100 * u.TeV, 3)
    actual = SkyCube.empty_like(reference=image, energies=energies)

    desired = SkyCube.empty(nxpix=11, nypix=7, enumbins=3, mode='center',
                            emin=1, emax=100, eunit='TeV')
    SkyCube.assert_allclose(actual, desired)


@requires_dependency('scipy')
def test_empty_like_energy_bounds():
    image = SkyImage.empty(nxpix=11, nypix=7)
    energies = EnergyBounds.equal_log_spacing(1 * u.TeV, 100 * u.TeV, 4)
    actual = SkyCube.empty_like(reference=image, energies=energies)

    desired = SkyCube.empty(nxpix=11, nypix=7, enumbins=4, mode='edges',
                            emin=1, emax=100, eunit='TeV')
    SkyCube.assert_allclose(actual, desired)


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
        energy = self.sky_cube.energies()

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

    def test_str(self):
        actual = str(self.sky_cube)
        expected = textwrap.dedent("""\
        Sky cube flux with shape=(30, 21, 61) and unit=1 / (cm2 MeV s sr):
         n_lon:       61  type_lon:    GLON-CAR         unit_lon:    deg
         n_lat:       21  type_lat:    GLAT-CAR         unit_lat:    deg
         n_energy:    30  unit_energy: MeV
        """)
        assert actual == expected

    @requires_dependency('scipy')
    @requires_dependency('regions')
    def test_spectrum(self):
        center = SkyCoord(0, 0, frame='galactic', unit='deg')
        radius = 1 * u.deg
        region = CircleSkyRegion(center, radius)
        spectrum = self.sky_cube.spectrum(region)

        assert_quantity_allclose(spectrum['e_ref'].quantity,
                                 self.sky_cube.energies('center'))

        assert_quantity_allclose(spectrum['e_min'].quantity,
                                 self.sky_cube.energies('edges')[:-1])

        assert_quantity_allclose(spectrum['e_max'].quantity,
                                 self.sky_cube.energies('edges')[1:])

        assert_quantity_allclose(spectrum['value'].quantity.sum(),
                                 8.710522670298815E-4 * u.Unit('1 / (cm2 MeV s sr)'))


@requires_dependency('scipy')
class TestSkyCubeInterpolation(object):
    def setup_class(self):
        self.sky_cube = make_test_sky_cube()
        self.pwl = make_test_spectral_model()

    def test_sky_image(self):
        energy = 50 * u.TeV
        image = self.sky_cube.sky_image(energy, interpolation='linear')
        assert_quantity_allclose(image.data, self.pwl(energy))

    def test_sky_image_integrate(self):
        emin, emax = [1, 100] * u.TeV
        integral = self.sky_cube.sky_image_integral(emin, emax)
        assert_quantity_allclose(integral.data, self.pwl.integral(emin, emax))

    def test_bin_size(self):
        bin_size = self.sky_cube.bin_size
        assert bin_size.shape == (4, 3, 3)
        assert bin_size.unit == 'TeV sr'

        assert_allclose(bin_size.value[0, 0, 0], 2.6346694056569e-07)
        assert_allclose(bin_size.value.sum(), 0.00010856564234889744)

    @requires_dependency('reproject')
    def test_reproject(self):
        emin = 1 * u.TeV
        emax = 100 * u.TeV
        ref = SkyCube.empty(
            emin=emin, emax=emax, enumbins=4,
            nxpix=6, nypix=6, binsz=0.01,
        )
        reprojected = self.sky_cube.reproject(ref)

        # Check if reprojection conserves total flux
        integral = self.sky_cube.sky_image_integral(emin, emax)
        flux = (integral.data * integral.solid_angle()).sum()

        integral_rep = reprojected.sky_image_integral(emin, emax)
        flux_rep = (integral_rep.data * integral_rep.solid_angle()).sum()

        assert_quantity_allclose(flux, flux_rep)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_bin_events_in_cube():
    filename = ('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599'
                '/run023523/hess_events_023523.fits.gz')
    events = EventList.read(filename)
    meta = events.table.meta
    counts = SkyCube.empty(
        emin=0.5, emax=80, enumbins=8, eunit='TeV',
        dtype='int', nxpix=200, nypix=200, mode='edges',
        xref=meta['RA_OBJ'], yref=meta['DEC_OBJ'], coordsys='CEL',
    )

    counts.fill_events(events)

    # check against event list energy selection
    counts_image = SkyImage.empty(dtype='int', nxpix=200, nypix=200, xref=meta['RA_OBJ'],
                                  yref=meta['DEC_OBJ'], coordsys='CEL', proj='CAR')
    events = events.select_energy([0.5, 80] * u.TeV)
    counts_image.fill_events(events)

    assert counts.data.sum() == 1233
    assert counts.data.sum() == counts_image.data.sum()


@requires_dependency('scipy')
def test_conversion_wcs_map_nd():
    """Check conversion SkyCube <-> WCSMapNd"""
    cube = make_test_sky_cube()
    # TODO: add unit back once it's properly supported
    cube.data = cube.data.value

    map = cube.to_wcs_nd_map()
    cube2 = SkyCube.from_wcs_nd_map(map)
    SkyCube.assert_allclose(cube, cube2)

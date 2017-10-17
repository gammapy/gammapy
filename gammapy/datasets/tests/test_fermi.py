# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing.utils import assert_allclose
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest
from astropy.units import Quantity
from astropy.coordinates import Angle

from ...utils.testing import requires_dependency, requires_data
from ...datasets import (
    FermiLATDataset,
    FermiGalacticCenter,
    FermiVelaRegion,
    load_lat_psf_performance,
)


@requires_data('gammapy-extra')
class TestFermiGalacticCenter:
    def test_filenames(self):
        filenames = FermiGalacticCenter.filenames()
        assert isinstance(filenames, dict)

    @requires_dependency('scipy')
    def test_psf(self):
        psf = FermiGalacticCenter.psf()
        energy = Quantity(110, 'GeV')
        fraction = 0.68
        interpol_param = dict(method='nearest', bounds_error=False)
        angle = psf.containment_radius(energy, fraction, interpol_param)
        assert_quantity_allclose(angle, Angle(0.1927459865412511, 'deg'), rtol=1e-2)

    def test_counts(self):
        counts = FermiGalacticCenter.counts()
        assert counts.data.shape == (201, 401)
        assert counts.data.sum() == 24803

    @requires_dependency('scipy')
    def test_diffuse_model(self):
        diffuse_model = FermiGalacticCenter.diffuse_model()
        assert diffuse_model.data.shape == (30, 21, 61)
        assert_quantity_allclose(diffuse_model.energies()[0], Quantity(50, 'MeV'))

    @requires_dependency('scipy')
    def test_exposure_cube(self):
        exposure_cube = FermiGalacticCenter.exposure_cube()
        assert exposure_cube.data.shape == (21, 11, 31)
        assert_quantity_allclose(exposure_cube.energies()[0], Quantity(50, 'MeV'))


@requires_data('gammapy-extra')
class TestFermiVelaRegion:
    def test_filenames(self):
        filenames = FermiVelaRegion.filenames()
        assert isinstance(filenames, dict)

    def test_counts_cube(self):
        counts = FermiVelaRegion.counts_cube()[0]
        assert counts.data.shape == (20, 50, 50)
        assert counts.data.sum() == 1551

    @requires_dependency('scipy')
    def test_psf(self):
        psf = FermiVelaRegion.psf()
        energy = Quantity(110, 'GeV')
        fraction = 0.68
        interpol_param = dict(method='nearest', bounds_error=False)
        angle = psf.containment_radius(energy, fraction, interpol_param)
        assert_quantity_allclose(angle, Angle(0.13185321269896136, 'deg'), rtol=1e-1)

    @requires_dependency('scipy')
    def test_diffuse_model(self):
        diffuse_model = FermiVelaRegion.diffuse_model()
        assert diffuse_model.data.shape == (30, 161, 161)

    def test_background_image(self):
        background = FermiVelaRegion.background_image()
        assert background.data.shape == (50, 50)
        assert background.data.sum(), 264.54391

    def test_predicted_image(self):
        background = FermiVelaRegion.predicted_image()
        assert background.data.shape == (50, 50)
        assert background.data.sum(), 322.12299

    def test_events(self):
        events_list = FermiVelaRegion.events()
        assert events_list['EVENTS'].data.shape == (2042,)

    @requires_dependency('scipy')
    def test_exposure_cube(self):
        exposure_cube = FermiVelaRegion.exposure_cube()
        assert exposure_cube.data.shape == (21, 50, 50)
        assert exposure_cube.data.value.sum(), 4.978616e+15
        assert_quantity_allclose(exposure_cube.energies()[0], Quantity(10000, 'MeV'))

    def test_livetime(self):
        livetime_list = FermiVelaRegion.livetime_cube()
        assert livetime_list[1].data.shape == (12288,)
        assert livetime_list[2].data.shape == (12288,)
        assert livetime_list[3].data.shape == (4,)
        assert livetime_list[4].data.shape == (17276,)


@requires_data('gammapy-extra')
def test_load_lat_psf_performance():
    """Tests loading of each file by asserting first value is correct."""

    table_p7rep_68 = load_lat_psf_performance('P7REP_SOURCE_V15_68')
    assert table_p7rep_68['energy'][0] == 29.65100879793481
    assert table_p7rep_68['containment_angle'][0] == 11.723606254043286

    table_p7rep_95 = load_lat_psf_performance('P7REP_SOURCE_V15_95')
    assert table_p7rep_95['energy'][0] == 29.989807922064667
    assert table_p7rep_95['containment_angle'][0] == 24.31544392270023

    table_p7_68 = load_lat_psf_performance('P7SOURCEV6_68')
    assert table_p7_68['energy'][0] == 31.9853049046
    assert table_p7_68['containment_angle'][0] == 14.7338699328

    table_p7_95 = load_lat_psf_performance('P7SOURCEV6_95')
    assert table_p7_95['energy'][0] == 31.6227766017
    assert table_p7_95['containment_angle'][0] == 38.3847234362

@requires_data('fermi-lat')
@requires_dependency('healpy')
@requires_dependency('yaml')
class TestFermiLATDataset:
    def setup(self):
        filename = '$GAMMAPY_FERMI_LAT_DATA/2fhl/fermi_2fhl_data_config.yaml'
        self.data_2fhl = FermiLATDataset(filename)

    def test_events(self):
        events = self.data_2fhl.events
        assert len(events.table) == 60275

    def test_exposure(self):
        exposure = self.data_2fhl.exposure
        assert_allclose(exposure.data.sum(), 6.072634932461568e+16)

    def test_counts(self):
        counts = self.data_2fhl.counts
        assert_allclose(counts.data.sum(), 60275)

    def test_psf(self):
        psf = self.data_2fhl.psf
        actual = self.data_2fhl.psf.integral(100 * u.GeV, 0 * u.deg, 2 * u.deg)
        desired = 1.00048
        assert_allclose(actual, desired, rtol=1e-4)

    def test_isodiff(self):
        isodiff = self.data_2fhl.isotropic_diffuse
        actual = isodiff(10 * u.GeV)
        desired = 6.3191722098744606e-12 * u.Unit('1 / (cm2 MeV s sr)')
        assert_quantity_allclose(actual, desired)

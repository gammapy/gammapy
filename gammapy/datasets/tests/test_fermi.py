# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing.utils import assert_allclose
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import Angle
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...datasets import (
    FermiLATDataset,
    FermiGalacticCenter,
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
        assert_allclose(exposure.data.sum(), 6.072634932461568e+16, rtol=1e-5)

    def test_counts(self):
        counts = self.data_2fhl.counts
        assert_allclose(counts.data.sum(), 60275)

    def test_psf(self):
        actual = self.data_2fhl.psf.integral(100 * u.GeV, 0 * u.deg, 2 * u.deg)
        desired = 1.00048
        assert_allclose(actual, desired, rtol=1e-4)

    def test_isodiff(self):
        isodiff = self.data_2fhl.isotropic_diffuse
        actual = isodiff(10 * u.GeV)
        desired = 6.3191722098744606e-12 * u.Unit('1 / (cm2 MeV s sr)')
        assert_quantity_allclose(actual, desired)

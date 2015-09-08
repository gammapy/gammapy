# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.tests.helper import remote_data
from ...datasets import (FermiGalacticCenter,
                         FermiVelaRegion,
                         fetch_fermi_catalog,
                         fetch_fermi_extended_sources,
                         fetch_fermi_diffuse_background_model,
                         load_lat_psf_performance,
                         )

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestFermiGalacticCenter():
    def test_filenames(self):
        filenames = FermiGalacticCenter.filenames()
        assert isinstance(filenames, dict)

    @pytest.mark.skipif('not HAS_SCIPY')
    def test_psf(self):
        psf = FermiGalacticCenter.psf()
        energy = Quantity(100, 'GeV')
        fraction = 0.68
        angle = psf.containment_radius(energy, fraction)
        assert_quantity_allclose(angle, Angle(0.1927459865412511, 'degree'))

    def test_counts(self):
        counts = FermiGalacticCenter.counts()
        assert counts.data.shape == (201, 401)
        assert counts.data.sum() == 24803

    def test_diffuse_model(self):
        diffuse_model = FermiGalacticCenter.diffuse_model()
        assert diffuse_model.data.shape == (30, 21, 61)
        assert_quantity_allclose(diffuse_model.energy[0], Quantity(50, 'MeV'))

    def test_exposure_cube(self):
        exposure_cube = FermiGalacticCenter.exposure_cube()
        assert exposure_cube.data.shape == (21, 11, 31)
        assert_quantity_allclose(exposure_cube.energy[0], Quantity(50, 'MeV'))


class TestFermiVelaRegion():
    @remote_data
    def test_filenames(self):
        filenames = FermiVelaRegion.filenames()
        assert isinstance(filenames, dict)

    @remote_data
    def test_counts_cube(self):
        counts = FermiVelaRegion.counts_cube()[0]
        assert counts.data.shape == (20, 50, 50)
        assert counts.data.sum() == 1551

    @remote_data
    @pytest.mark.skipif('not HAS_SCIPY')
    def test_psf(self):
        psf = FermiVelaRegion.psf()
        energy = Quantity(100, 'GeV')
        fraction = 0.68
        angle = psf.containment_radius(energy, fraction)
        assert_quantity_allclose(angle, Angle(0.13185321269896136, 'degree'))

    @remote_data
    def test_diffuse_model(self):
        diffuse_model = FermiVelaRegion.diffuse_model()
        assert diffuse_model.data.shape == (30, 161, 161)

    @remote_data
    def test_background_image(self):
        background = FermiVelaRegion.background_image()
        assert background.data.shape == (50, 50)
        assert background.data.sum(), 264.54391

    @remote_data
    def test_predicted_image(self):
        background = FermiVelaRegion.predicted_image()
        assert background.data.shape == (50, 50)
        assert background.data.sum(), 322.12299

    @remote_data
    def test_events(self):
        events_list = FermiVelaRegion.events()
        assert events_list['EVENTS'].data.shape == (2042,)

    @remote_data
    def test_exposure_cube(self):
        exposure_cube = FermiVelaRegion.exposure_cube()
        assert exposure_cube.data.shape == (21, 50, 50)
        assert exposure_cube.data.value.sum(), 4.978616e+15
        assert_quantity_allclose(exposure_cube.energy[0], Quantity(10000, 'MeV'))

    @remote_data
    def test_livetime(self):
        livetime_list = FermiVelaRegion.livetime_cube()
        assert livetime_list[1].data.shape == (12288,)
        assert livetime_list[2].data.shape == (12288,)
        assert livetime_list[3].data.shape == (4,)
        assert livetime_list[4].data.shape == (17276,)


@remote_data
def test_fetch_fermi_catalog():
    n_hdu = len(fetch_fermi_catalog('3FGL'))
    assert n_hdu == 6

    n_sources = len(fetch_fermi_catalog('3FGL', 'LAT_Point_Source_Catalog'))
    assert n_sources == 3034

    n_hdu = len(fetch_fermi_catalog('2FGL'))
    assert n_hdu == 5

    n_sources = len(fetch_fermi_catalog('2FGL', 'LAT_Point_Source_Catalog'))
    assert n_sources == 1873


@remote_data
def test_fetch_fermi_extended_sources():
    assert len(fetch_fermi_extended_sources('3FGL')) == 26
    assert len(fetch_fermi_extended_sources('2FGL')) == 12
    assert len(fetch_fermi_extended_sources('1FHL')) == 23


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

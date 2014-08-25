# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.tests.helper import remote_data
from ...utils.testing import assert_quantity
from ...datasets import (FermiGalacticCenter,
                         FermiVelaRegion,
                         fetch_fermi_catalog,
                         fetch_fermi_extended_sources,
                         fetch_fermi_diffuse_background_model,
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
        assert_quantity(angle, Angle('0.1927459865412511 deg'))

    def test_counts(self):
        counts = FermiGalacticCenter.counts()
        assert counts.data.shape == (201, 401)
        assert counts.data.sum() == 24803

    def test_diffuse_model(self):
        diffuse_model = FermiGalacticCenter.diffuse_model()
        assert diffuse_model.data.shape == (30, 21, 61)
        assert_quantity(diffuse_model.energy[0], Quantity(50, 'MeV'))

    def test_exposure_cube(self):
        exposure_cube = FermiGalacticCenter.exposure_cube()
        assert exposure_cube.data.shape == (21, 11, 31)
        assert_quantity(exposure_cube.energy[0], Quantity(50, 'MeV'))


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
        assert_quantity(angle, Angle('0.13185321269896136 deg'))

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
        assert_quantity(exposure_cube.energy[0], Quantity(10000, 'MeV'))

    @remote_data
    def test_livetime(self):
        livetime_list = FermiVelaRegion.livetime_cube()
        assert livetime_list[1].data.shape == (12288,)
        assert livetime_list[2].data.shape == (12288,)
        assert livetime_list[3].data.shape == (4,)
        assert livetime_list[4].data.shape == (17276,)


@remote_data
def test_fetch_fermi_catalog():
    n_hdu = len(fetch_fermi_catalog('2FGL'))
    assert n_hdu == 5

    n_sources = len(fetch_fermi_catalog('2FGL', 'LAT_Point_Source_Catalog'))
    assert n_sources == 1873


@remote_data
def test_fetch_fermi_extended_sources():
    assert len(fetch_fermi_extended_sources('2FGL')) == 12
    assert len(fetch_fermi_extended_sources('1FHL')) == 23

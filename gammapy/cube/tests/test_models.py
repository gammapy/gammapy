# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from ...utils.testing import requires_dependency
from ...maps import MapAxis, WcsGeom, Map
from ...image.models import SkyGaussian
from ...spectrum.models import PowerLaw
from ..models import SkyModel, SkyModelMapEvaluator


@pytest.fixture(scope='session')
def geom():
    axis = MapAxis.from_edges(np.logspace(-1, 1, 3), unit=u.TeV)
    return WcsGeom.create(skydir=(0, 0), npix=(5, 4), coordsys='GAL', axes=[axis])


@pytest.fixture(scope='session')
def exposure(geom):
    m = Map.from_geom(geom)
    m.quantity = np.ones((2, 4, 5)) * u.Quantity('100 m2 s')
    return m

@pytest.fixture(scope='session')
def background(geom):
    m = Map.from_geom(geom)
    m.quantity = np.ones((2, 4, 5))*1e-7
    return m


@pytest.fixture(scope='session')
def sky_model():
    spatial_model = SkyGaussian(
        lon_0='3 deg', lat_0='4 deg', sigma='3 deg',
    )
    spectral_model = PowerLaw(
        index=2, amplitude='1e-11 cm-2 s-1 TeV-1', reference='1 TeV',
    )
    return SkyModel(spatial_model, spectral_model)


class TestSkyModel:
    @staticmethod
    def test_repr(sky_model):
        assert 'SkyModel' in repr(sky_model)

    @staticmethod
    def test_str(sky_model):
        assert 'SkyModel' in str(sky_model)

    @staticmethod
    def test_evaluate_scalar(sky_model):
        lon = 3 * u.deg
        lat = 4 * u.deg
        energy = 1 * u.TeV

        q = sky_model.evaluate(lon, lat, energy)

        assert q.unit == 'cm-2 s-1 TeV-1 deg-2'
        assert q.shape == (1, 1, 1)
        assert_allclose(q.value, 1.76838826e-13)

    @staticmethod
    def test_evaluate_array(sky_model):
        lon = 3 * u.deg * np.ones(shape=(3, 4))
        lat = 4 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV

        q = sky_model.evaluate(lon, lat, energy)

        assert q.shape == (5, 3, 4)
        assert_allclose(q.value, 1.76838826e-13)


@requires_dependency('scipy')
class TestSkyModelMapEvaluator:

    def setup(self):
        self.evaluator = SkyModelMapEvaluator(sky_model(), exposure(geom()))

    def test_energy_center(self):
        val = self.evaluator.energy_center
        assert val.shape == (2,)
        assert val.unit == 'TeV'

    def test_energy_edges(self):
        val = self.evaluator.energy_edges
        assert val.shape == (3,)
        assert val.unit == 'TeV'

    def test_energy_bin_width(self):
        val = self.evaluator.energy_bin_width
        assert val.shape == (2,)
        assert val.unit == 'TeV'

    def test_lon_lat(self):
        val = self.evaluator.lon
        assert val.shape == (4, 5)
        assert val.unit == 'deg'

        val = self.evaluator.lat
        assert val.shape == (4, 5)
        assert val.unit == 'deg'

    def test_solid_angle(self):
        val = self.evaluator.solid_angle
        assert val.shape == (2, 4, 5)
        assert val.unit == 'sr'

    def test_bin_volume(self):
        val = self.evaluator.bin_volume
        assert val.shape == (2, 4, 5)
        assert val.unit == 'TeV sr'

    def test_compute_dnde(self):
        out = self.evaluator.compute_dnde()

        assert out.shape == (2, 4, 5)
        assert out.unit == 'cm-2 s-1 TeV-1 deg-2'
        assert_allclose(out.value.mean(), 7.460919e-14)

    def test_compute_flux(self):
        out = self.evaluator.compute_flux()

        assert out.shape == (2, 4, 5)
        assert out.unit == 'cm-2 s-1'
        assert_allclose(out.value.mean(), 1.828206748668197e-14)

    def test_compute_npred(self):
        out = self.evaluator.compute_npred()
        assert out.shape == (2, 4, 5)
        npred_expected = 7.312826994672788e-07
        assert_allclose(out.sum(), npred_expected)

        evaluator_bkg = SkyModelMapEvaluator(sky_model(), exposure(geom()), background=background(geom()))
        out_bkg = evaluator_bkg.compute_npred()
        npred_back_expected = npred_expected + 40.0e-7
        assert_allclose(out_bkg.sum(), npred_back_expected)



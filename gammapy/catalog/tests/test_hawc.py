# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from gammapy.catalog import SourceCatalog2HWC
from gammapy.utils.testing import requires_data, requires_dependency
from gammapy.modeling.models import (
    DiskSpatialModel,
    PointSpatialModel,
    PowerLawSpectralModel,
)


@pytest.fixture(scope="session")
def cat():
    return SourceCatalog2HWC()


@requires_data()
class TestSourceCatalog2HWC:
    @staticmethod
    def test_source_table(cat):
        assert cat.name == "2hwc"
        assert len(cat.table) == 40

    @staticmethod
    def test_positions(cat):
        assert len(cat.positions) == 40


@requires_data()
class TestSourceCatalogObject2HWC:
    @staticmethod
    def test_data(cat):
        assert cat[0].data["source_name"] == "2HWC J0534+220"
        assert cat[0].n_models == 1

        assert cat[1].data["source_name"] == "2HWC J0631+169"
        assert cat[1].n_models == 2

    @staticmethod
    def test_str(cat):
        expected = open(get_pkg_data_filename("data/2hwc_j0534+220.txt")).read()
        assert str(cat[0]) == expected

        expected = open(get_pkg_data_filename("data/2hwc_j0631+169.txt")).read()
        assert str(cat[1]) == expected

    @staticmethod
    def test_position(cat):
        position = cat[0].position
        assert_allclose(position.ra.deg, 83.628, atol=1e-3)
        assert_allclose(position.dec.deg, 22.024, atol=1e-3)

    @staticmethod
    def test_sky_model(cat):
        model = cat[1].sky_model("extended")
        assert model.name == "2HWC J0631+169"
        assert isinstance(model.spectral_model, PowerLawSpectralModel)
        assert isinstance(model.spatial_model, DiskSpatialModel)

        with pytest.raises(ValueError):
            cat[0].sky_model("extended")

    @staticmethod
    @requires_dependency("uncertainties")
    def test_spectral_model(cat):
        m = cat[0].spectral_model()
        flux, flux_err = m.integral_error(1 * u.TeV, 10 * u.TeV)
        assert flux.unit == "cm-2 s-1"
        assert_allclose(flux.value, 1.72699e-11, rtol=1e-3)
        assert flux_err.unit == "cm-2 s-1"
        assert_allclose(flux_err.value, 3.252178e-13, rtol=1e-3)

    @staticmethod
    @requires_dependency("uncertainties")
    def test_spatial_model(cat):
        m = cat[1].spatial_model()
        # p = m.parameters

        assert isinstance(m, PointSpatialModel)
        assert m.lon_0.unit == "deg"
        assert_allclose(m.lon_0.value, 195.614, atol=1e-2)
        # TODO: add assert on position error
        # assert_allclose(p.error("lon_0"), tbd)
        assert m.lat_0.unit == "deg"
        assert_allclose(m.lat_0.value, 3.507, atol=1e-2)
        assert m.frame == "galactic"

        m = cat[1].spatial_model("extended")

        assert isinstance(m, DiskSpatialModel)
        assert m.lon_0.unit == "deg"
        assert_allclose(m.lon_0.value, 195.614, atol=1e-10)
        assert m.lat_0.unit == "deg"
        assert_allclose(m.lat_0.value, 3.507, atol=1e-10)
        assert m.frame == "galactic"
        assert m.r_0.unit == "deg"
        assert_allclose(m.r_0.value, 2.0, atol=1e-3)

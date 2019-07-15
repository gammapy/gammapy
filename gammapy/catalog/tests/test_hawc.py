# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from ...utils.testing import requires_data, requires_dependency
from ..hawc import SourceCatalog2HWC


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
        source = cat[0]
        assert source.data["source_name"] == "2HWC J0534+220"

    @staticmethod
    def test_position(cat):
        position = cat[0].position
        assert_allclose(position.ra.deg, 83.628, atol=1e-3)
        assert_allclose(position.dec.deg, 22.024, atol=1e-3)

    @staticmethod
    def test_str(cat):
        source = cat[0]
        assert "2HWC J0534+220" in str(source)
        assert "No second spectrum available for this source" in str(source)

        source = cat[1]
        assert "2HWC J0631+169" in str(source)
        assert "Spectrum 1:" in str(source)

    @staticmethod
    @requires_dependency("uncertainties")
    def test_spectral_models_one(cat):
        source = cat[0]
        assert source.n_spectra == 1

        spectral_models = source.spectral_models
        assert len(spectral_models) == 1

        e_min, e_max = [1, 10] * u.TeV
        flux, flux_err = spectral_models[0].integral_error(e_min, e_max)
        assert flux.unit == "cm-2 s-1"
        assert_allclose(flux.value, 1.2966462620662674e-12)
        assert flux_err.unit == "cm-2 s-1"
        assert_allclose(flux_err.value, 1.671177271712936e-14)

    @staticmethod
    @requires_dependency("uncertainties")
    def test_spectral_models_two(cat):
        # This test is just to check that sources with 2 spectra also work OK.
        source = cat[1]
        assert source.n_spectra == 2

        spectral_models = source.spectral_models
        assert len(spectral_models) == 2

        e_min, e_max = [1, 10] * u.TeV
        flux, flux_err = spectral_models[1].integral_error(e_min, e_max)
        assert flux.unit == "cm-2 s-1"
        assert_allclose(flux.value, 3.3381204455973463e-13)
        assert flux_err.unit == "cm-2 s-1"
        assert_allclose(flux_err.value, 4.697084075095061e-14)

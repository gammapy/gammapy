# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from ...utils.testing import requires_data, requires_dependency
from ..hawc import SourceCatalog2HWC

# 2HWC catalog is in ECSV format, which requires yaml to read the header
pytest.importorskip("yaml")


@pytest.fixture(scope="session")
def hawc_2hwc():
    return SourceCatalog2HWC()


@requires_data("gammapy-extra")
class TestSourceCatalog2HWC:
    def test_source_table(self, hawc_2hwc):
        assert hawc_2hwc.name == "2hwc"
        assert len(hawc_2hwc.table) == 40


@requires_data("gammapy-extra")
class TestSourceCatalogObject2HWC:
    def test_data(self, hawc_2hwc):
        source = hawc_2hwc[0]
        assert source.data["source_name"] == "2HWC J0534+220"

    def test_str(self, hawc_2hwc):
        source = hawc_2hwc[0]
        assert "2HWC J0534+220" in str(source)
        assert "No second spectrum available for this source" in str(source)

        source = hawc_2hwc[1]
        assert "2HWC J0631+169" in str(source)
        assert "Spectrum 1:" in str(source)

    @requires_dependency("uncertainties")
    def test_spectral_models_one(self, hawc_2hwc):
        source = hawc_2hwc[0]
        assert source.n_spectra == 1

        spectral_models = source.spectral_models
        assert len(spectral_models) == 1

        e_min, e_max = [1, 10] * u.TeV
        flux, flux_err = spectral_models[0].integral_error(e_min, e_max)
        assert flux.unit == u.Unit("cm-2 s-1")
        assert_allclose(flux.value, 1.2966462620662674e-12)
        assert flux_err.unit == u.Unit("cm-2 s-1")
        assert_allclose(flux_err.value, 1.671177271712936e-14)

    @requires_dependency("uncertainties")
    def test_spectral_models_two(self, hawc_2hwc):
        # This test is just to check that sources with 2 spectra also work OK.
        source = hawc_2hwc[1]
        assert source.n_spectra == 2

        spectral_models = source.spectral_models
        assert len(spectral_models) == 2

        e_min, e_max = [1, 10] * u.TeV
        flux, flux_err = spectral_models[1].integral_error(e_min, e_max)
        assert flux.unit == u.Unit("cm-2 s-1")
        assert_allclose(flux.value, 3.3381204455973463e-13)
        assert flux_err.unit == u.Unit("cm-2 s-1")
        assert_allclose(flux_err.value, 4.697084075095061e-14)

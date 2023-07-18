# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.catalog import SourceCatalog1LHAASO
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PointSpatialModel,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    TemplateSpatialModel,
)
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def lhaaso1():
    return SourceCatalog1LHAASO()


@requires_data()
class TestSourceCatalog1LHAASO:
    @staticmethod
    def test_source_table(lhaaso1):
        assert lhaaso1.tag == "1LHAASO"
        assert len(lhaaso1.table) == 90

    @staticmethod
    def test_positions(lhaaso1):
        assert len(lhaaso1.positions) == 90

    @staticmethod
    def test_to_models(lhaaso1):
        models = lhaaso1.to_models(which="both")
        assert len(models) == 90

        models = lhaaso1.to_models(which="KM2A")
        assert np.all(
            [m.spectral_model.reference.quantity == 50 * u.TeV for m in models]
        )
        assert len(models) == 75
        models = lhaaso1.to_models(which="WCDA")
        assert np.all(
            [m.spectral_model.reference.quantity == 3 * u.TeV for m in models]
        )
        assert len(models) == 69


@requires_data()
class TestSourceCatalogObject1LHAASO:
    @staticmethod
    def test_data(lhaaso1):
        assert lhaaso1[0].data["Source_Name"] == "1LHAASO J0007+5659u"
        assert "KM2A" in lhaaso1[0].data["Model_a"]

        assert_allclose(lhaaso1[0].data["r39_ul"].value, 0.18)
        assert lhaaso1[0].data["r39_ul"].unit == u.deg

        assert_allclose(lhaaso1[0].data["N0"].value, 0.33e-16)
        assert lhaaso1[0].data["N0"].unit == u.Unit("cm−2 s−1 TeV−1")

        assert_allclose(lhaaso1[0].data["N0_err"].value, 0.05e-16)
        assert lhaaso1[0].data["N0_err"].unit == u.Unit("cm−2 s−1 TeV−1")

        assert_allclose(lhaaso1[0].data["N0_ul_b"].value, 0.27e-13)
        assert lhaaso1[0].data["N0_ul_b"].unit == u.Unit("cm−2 s−1 TeV−1")

        assert lhaaso1[1].data["ASSO_Name"] == "CTA 1"
        assert_allclose(lhaaso1[1].data["ASSO_Sep"].value, 0.12)
        assert lhaaso1[0].data["ASSO_Sep"].unit == u.deg

        assert lhaaso1[10].data["Source_Name"] == "1LHAASO J0428+5531*"
        assert "WCDA" in lhaaso1[10].data["Model_a"]

        assert_allclose(lhaaso1[10].data["RAJ2000"].value, 67.23)
        assert_allclose(lhaaso1[10].data["DECJ2000"].value, 55.53)
        assert_allclose(lhaaso1[10].data["pos_err"].value, 0.36)

        assert lhaaso1[10].data["RAJ2000"].unit == u.deg
        assert lhaaso1[10].data["DECJ2000"].unit == u.deg
        assert lhaaso1[10].data["pos_err"].unit == u.deg

        assert_allclose(lhaaso1[10].data["r39"].value, 1.18)
        assert_allclose(lhaaso1[10].data["r39_b"].value, 0.32)
        assert lhaaso1[10].data["r39_b"].unit == u.deg

        assert_allclose(lhaaso1[10].data["r39_err"].value, 0.12)
        assert_allclose(lhaaso1[10].data["r39_err_b"].value, 0.06)
        assert lhaaso1[10].data["r39_err_b"].unit == u.deg

    @staticmethod
    def test_position(lhaaso1):
        position = lhaaso1[0].position
        assert_allclose(position.ra.deg, 1.86, atol=1e-3)
        assert_allclose(position.dec.deg, 57.00, atol=1e-3)

    @staticmethod
    def test_sky_model(lhaaso1):
        model = lhaaso1[0].sky_model("both")
        assert model.name == "1LHAASO J0007+5659u"
        assert isinstance(model.spectral_model, PowerLawSpectralModel)
        assert isinstance(model.spatial_model, PointSpatialModel)

        assert lhaaso1[0].sky_model("WCDA") is None

        model = lhaaso1[1].sky_model("both")
        assert model.name == "1LHAASO J0007+7303u"
        assert isinstance(model.spectral_model, PowerLawNormSpectralModel)
        assert isinstance(model.spatial_model, TemplateSpatialModel)

        model = lhaaso1[1].sky_model("KM2A")
        assert model.name == "1LHAASO J0007+7303u"
        assert isinstance(model.spectral_model, PowerLawSpectralModel)
        assert isinstance(model.spatial_model, GaussianSpatialModel)

        model = lhaaso1[1].sky_model("WCDA")
        assert model.name == "1LHAASO J0007+7303u"
        assert isinstance(model.spectral_model, PowerLawSpectralModel)
        assert isinstance(model.spatial_model, PointSpatialModel)

        model = lhaaso1[11].sky_model("both")
        assert model.name == "1LHAASO J0500+4454"
        assert isinstance(model.spectral_model, PowerLawSpectralModel)
        assert isinstance(model.spatial_model, GaussianSpatialModel)

    @staticmethod
    def test_spectral_model(lhaaso1):
        m = lhaaso1[0].spectral_model("KM2A")
        dnde, dnde_err = m.evaluate_error(50 * u.TeV)
        assert dnde.unit == "cm-2 s-1 TeV-1"
        assert_allclose(dnde.value, 0.33e-16, rtol=1e-3)
        assert_allclose(dnde_err.value, 0.05e-16, rtol=1e-3)

        m = lhaaso1[11].spectral_model("WCDA")
        dnde, dnde_err = m.evaluate_error(3 * u.TeV)
        assert dnde.unit == "cm-2 s-1 TeV-1"
        assert_allclose(dnde.value, 0.69e-13, rtol=1e-3)
        assert_allclose(dnde_err.value, 0.16e-13, rtol=1e-3)

    @staticmethod
    def test_spatial_model(lhaaso1):
        m = lhaaso1[0].spatial_model("KM2A")

        assert isinstance(m, PointSpatialModel)
        assert m.lon_0.unit == "deg"
        assert_allclose(m.lon_0.value, 1.86, atol=1e-2)
        assert_allclose(m.lon_0.error, 0.09, atol=1e-2)
        assert m.lat_0.unit == "deg"
        assert_allclose(m.lat_0.value, 57.00, atol=1e-2)
        assert_allclose(m.lat_0.error, 0.049, atol=1e-2)
        assert m.frame == "fk5"

        m = lhaaso1[11].spatial_model("WCDA")
        assert isinstance(m, GaussianSpatialModel)
        assert m.lon_0.unit == "deg"
        assert_allclose(m.lon_0.value, 75.01, atol=1e-10)
        assert m.lat_0.unit == "deg"
        assert_allclose(m.lat_0.value, 44.92, atol=1e-10)
        assert m.frame == "fk5"
        assert m.sigma.unit == "deg"
        assert_allclose(m.sigma.value, 0.41, atol=1e-3)

        model = lhaaso1["1LHAASO J0007+5659u"].spatial_model("KM2A")
        pos_err = model.position_error
        assert_allclose(pos_err.height.value, 2 * 0.12, rtol=1e-4)
        assert_allclose(pos_err.width.value, 2 * 0.12, rtol=1e-4)
        assert_allclose(model.position.ra.value, pos_err.center.ra.value)
        assert_allclose(model.position.dec.value, pos_err.center.dec.value)

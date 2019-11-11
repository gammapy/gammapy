# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.cube import MapEvaluator, PSFKernel
from gammapy.irf import EnergyDispersion
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    BackgroundModel,
    ConstantSpectralModel,
    GaussianSpatialModel,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyDiffuseCube,
    SkyModel,
    SkyModels,
    create_fermi_isotropic_diffuse_model,
)
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def sky_model():
    spatial_model = GaussianSpatialModel(
        lon_0="3 deg", lat_0="4 deg", sigma="3 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    return SkyModel(spatial_model, spectral_model, name="source-1")


@pytest.fixture(scope="session")
def diffuse_model():
    axis = MapAxis.from_nodes([0.1, 100], name="energy", unit="TeV", interp="log")
    m = Map.create(
        npix=(4, 3), binsz=2, axes=[axis], unit="cm-2 s-1 MeV-1 sr-1", coordsys="GAL"
    )
    m.data += 42
    return SkyDiffuseCube(m)


@pytest.fixture(scope="session")
def geom():
    axis = MapAxis.from_edges(np.logspace(-1, 1, 3), unit=u.TeV, name="energy")
    return WcsGeom.create(skydir=(0, 0), npix=(5, 4), coordsys="GAL", axes=[axis])


@pytest.fixture(scope="session")
def geom_true():
    axis = MapAxis.from_edges(np.logspace(-1, 1, 4), unit=u.TeV, name="energy")
    return WcsGeom.create(skydir=(0, 0), npix=(5, 4), coordsys="GAL", axes=[axis])


@pytest.fixture(scope="session")
def exposure(geom_true):
    m = Map.from_geom(geom_true)
    m.quantity = np.ones(geom_true.data_shape) * u.Quantity("100 m2 s")
    m.data[1] *= 10
    return m


@pytest.fixture(scope="session")
def background(geom):
    m = Map.from_geom(geom)
    m.quantity = np.ones(geom.data_shape) * 1e-7
    return m


@pytest.fixture(scope="session")
def edisp(geom, geom_true):
    e_reco = geom.get_axis_by_name("energy").edges
    e_true = geom_true.get_axis_by_name("energy").edges
    return EnergyDispersion.from_diagonal_response(e_true=e_true, e_reco=e_reco)


@pytest.fixture(scope="session")
def psf(geom_true):
    sigma = 0.5 * u.deg
    return PSFKernel.from_gauss(geom_true, sigma)


@pytest.fixture(scope="session")
def evaluator(sky_model, exposure, psf, edisp):
    return MapEvaluator(sky_model, exposure, psf=psf, edisp=edisp)


@pytest.fixture(scope="session")
def diffuse_evaluator(diffuse_model, exposure, psf, edisp):
    return MapEvaluator(diffuse_model, exposure, psf=psf, edisp=edisp)


@pytest.fixture(scope="session")
def sky_models(sky_model):
    sky_model_2 = sky_model.copy(name="source-2")
    sky_model_3 = sky_model.copy(name="source-3")
    return SkyModels([sky_model_2, sky_model_3])


@pytest.fixture(scope="session")
def sky_models_2(sky_model):
    sky_model_4 = sky_model.copy(name="source-4")
    sky_model_5 = sky_model.copy(name="source-5")
    return SkyModels([sky_model_4, sky_model_5])


def test_sky_model_init():
    with pytest.raises(TypeError):
        spatial_model = GaussianSpatialModel()
        SkyModel(spectral_model=1234, spatial_model=spatial_model)

    with pytest.raises(TypeError):
        SkyModel(spectral_model=PowerLawSpectralModel(), spatial_model=1234)


def test_skymodel_addition(sky_model, sky_models, sky_models_2, diffuse_model):
    models = sky_model + sky_model.copy()
    assert isinstance(models, SkyModels)
    assert len(models) == 2

    models = sky_model + sky_models
    assert isinstance(models, SkyModels)
    assert len(models) == 3

    models = sky_models + sky_model
    assert isinstance(models, SkyModels)
    assert len(models) == 3

    models = sky_models + diffuse_model
    assert isinstance(models, SkyModels)
    assert len(models) == 3

    models = sky_models + sky_models_2
    assert isinstance(models, SkyModels)
    assert len(models) == 4

    models = sky_model + sky_models
    assert isinstance(models, SkyModels)
    assert len(models) == 3


def test_background_model(background):
    bkg1 = BackgroundModel(background, norm=2.0).evaluate()
    assert_allclose(bkg1.data[0][0][0], background.data[0][0][0] * 2.0, rtol=1e-3)
    assert_allclose(bkg1.data.sum(), background.data.sum() * 2.0, rtol=1e-3)

    bkg2 = BackgroundModel(
        background, norm=2.0, tilt=0.2, reference="1000 GeV"
    ).evaluate()
    assert_allclose(bkg2.data[0][0][0], 2.254e-07, rtol=1e-3)
    assert_allclose(bkg2.data.sum(), 7.352e-06, rtol=1e-3)


class TestSkyModels:
    @staticmethod
    def test_parameters(sky_models):
        parnames = [
            "lon_0",
            "lat_0",
            "sigma",
            "e",
            "phi",
            "index",
            "amplitude",
            "reference",
        ] * 2
        assert sky_models.parameters.names == parnames

        # Check that model parameters are references to the parts
        p1 = sky_models.parameters["lon_0"]
        p2 = sky_models[0].parameters["lon_0"]
        assert p1 is p2

    @staticmethod
    def test_evaluate(sky_models):
        lon = 3 * u.deg * np.ones(shape=(3, 4))
        lat = 4 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV

        q = sky_models.evaluate(lon, lat, energy[:, np.newaxis, np.newaxis])

        assert q.unit == "cm-2 s-1 TeV-1 sr-1"
        assert q.shape == (5, 3, 4)
        assert_allclose(q.to_value("cm-2 s-1 TeV-1 deg-2"), 3.53758465e-13)

    @staticmethod
    def test_str(sky_models):
        assert "Component 0" in str(sky_models)
        assert "Component 1" in str(sky_models)

    @staticmethod
    def test_get_item(sky_models):
        model = sky_models["source-2"]
        assert model.name == "source-2"

        model = sky_models["source-3"]
        assert model.name == "source-3"

        with pytest.raises(IndexError):
            sky_models["spam"]


class TestSkyModel:
    @staticmethod
    def test_repr(sky_model):
        assert "SkyModel" in repr(sky_model)

    @staticmethod
    def test_str(sky_model):
        assert "SkyModel" in str(sky_model)

    @staticmethod
    def test_parameters(sky_model):
        # Check that model parameters are references to the spatial and spectral parts
        p1 = sky_model.parameters["lon_0"]
        p2 = sky_model.spatial_model.parameters["lon_0"]
        assert p1 is p2

        p1 = sky_model.parameters["amplitude"]
        p2 = sky_model.spectral_model.parameters["amplitude"]
        assert p1 is p2

    @staticmethod
    def test_evaluate_scalar(sky_model):
        lon = 3 * u.deg
        lat = 4 * u.deg
        energy = 1 * u.TeV

        q = sky_model.evaluate(lon, lat, energy)

        assert q.unit == "cm-2 s-1 TeV-1 sr-1"
        assert np.isscalar(q.value)
        assert_allclose(q.to_value("cm-2 s-1 TeV-1 deg-2"), 1.76879232e-13)

    @staticmethod
    def test_evaluate_array(sky_model):
        lon = 3 * u.deg * np.ones(shape=(3, 4))
        lat = 4 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV

        q = sky_model.evaluate(lon, lat, energy[:, np.newaxis, np.newaxis])

        assert q.shape == (5, 3, 4)
        assert_allclose(q.to_value("cm-2 s-1 TeV-1 deg-2"), 1.76879232e-13)


class TestSkyDiffuseCube:
    @staticmethod
    def test_evaluate_scalar(diffuse_model):
        # Check pixel inside map
        val = diffuse_model.evaluate(0 * u.deg, 0 * u.deg, 10 * u.TeV)
        assert val.unit == "cm-2 s-1 MeV-1 sr-1"
        assert val.shape == (1,)
        assert_allclose(val.value, 42)

        # Check pixel outside map (spatially)
        val = diffuse_model.evaluate(100 * u.deg, 0 * u.deg, 10 * u.TeV)
        assert_allclose(val.value, 0)

        # Check pixel outside energy range
        val = diffuse_model.evaluate(0 * u.deg, 0 * u.deg, 200 * u.TeV)
        assert_allclose(val.value, 0)

    @staticmethod
    def test_evaluate_array(diffuse_model):
        lon = 1 * u.deg * np.ones(shape=(3, 4))
        lat = 2 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV

        q = diffuse_model.evaluate(lon, lat, energy[:, np.newaxis, np.newaxis])

        assert q.shape == (5, 3, 4)
        assert_allclose(q.value.mean(), 42)

    @staticmethod
    @requires_data()
    def test_read():
        model = SkyDiffuseCube.read(
            "$GAMMAPY_DATA/tests/unbundled/fermi/gll_iem_v02_cutout.fits"
        )
        assert model.map.unit == "cm-2 s-1 MeV-1 sr-1"

        # Check pixel inside map
        val = model.evaluate(0 * u.deg, 0 * u.deg, 100 * u.GeV)
        assert val.unit == "cm-2 s-1 MeV-1 sr-1"
        assert val.shape == (1,)
        assert_allclose(val.value, 1.396424e-12, rtol=1e-5)

    @staticmethod
    def test_evaluation_radius(diffuse_model):
        radius = diffuse_model.evaluation_radius
        assert radius.unit == "deg"
        assert_allclose(radius.value, 4)

    @staticmethod
    def test_frame(diffuse_model):
        assert diffuse_model.frame == "galactic"


class TestSkyDiffuseCubeMapEvaluator:
    @staticmethod
    def test_compute_dnde(diffuse_evaluator):
        out = diffuse_evaluator.compute_dnde()
        assert out.shape == (3, 4, 5)
        out = out.to("cm-2 s-1 MeV-1 sr-1")
        assert_allclose(out.value.sum(), 2520.0, rtol=1e-5)
        assert_allclose(out.value[0, 0, 0], 42, rtol=1e-5)

    @staticmethod
    def test_compute_flux(diffuse_evaluator):
        out = diffuse_evaluator.compute_flux()
        assert out.shape == (3, 4, 5)
        out = out.to("cm-2 s-1")
        assert_allclose(out.value.sum(), 633263.444803, rtol=1e-5)
        assert_allclose(out.value[0, 0, 0], 1164.656176, rtol=1e-5)

    @staticmethod
    def test_apply_psf(diffuse_evaluator):
        flux = diffuse_evaluator.compute_flux()
        npred = diffuse_evaluator.apply_exposure(flux)
        out = diffuse_evaluator.apply_psf(npred)
        assert out.data.shape == (3, 4, 5)
        assert_allclose(out.data.sum(), 1.106404e12, rtol=1e-5)
        assert_allclose(out.data[0, 0, 0], 5.586508e08, rtol=1e-5)

    @staticmethod
    def test_apply_edisp(diffuse_evaluator):
        flux = diffuse_evaluator.compute_flux()
        npred = diffuse_evaluator.apply_exposure(flux)
        out = diffuse_evaluator.apply_edisp(npred)
        assert out.data.shape == (2, 4, 5)
        assert_allclose(out.data.sum(), 1.606345e12, rtol=1e-5)
        assert_allclose(out.data[0, 0, 0], 1.164656e09, rtol=1e-5)

    @staticmethod
    def test_compute_npred(diffuse_evaluator):
        out = diffuse_evaluator.compute_npred()
        assert out.data.shape == (2, 4, 5)
        assert_allclose(out.data.sum(), 1.106403e12, rtol=1e-5)
        assert_allclose(out.data[0, 0, 0], 5.586508e08, rtol=1e-5)


class TestSkyModelMapEvaluator:
    @staticmethod
    def test_compute_dnde(evaluator):
        out = evaluator.compute_dnde()
        assert out.shape == (3, 4, 5)
        assert out.unit == "cm-2 s-1 TeV-1 sr-1"
        assert_allclose(
            out.to_value("cm-2 s-1 TeV-1 deg-2").sum(),
            1.1788166328203174e-11,
            rtol=1e-5,
        )
        assert_allclose(
            out.to_value("cm-2 s-1 TeV-1 deg-2")[0, 0, 0],
            5.087056282039508e-13,
            rtol=1e-5,
        )

    @staticmethod
    def test_compute_flux(evaluator):
        out = evaluator.compute_flux().to_value("cm-2 s-1")
        assert out.shape == (3, 4, 5)
        assert_allclose(out.sum(), 1.291414e-12, rtol=1e-5)
        assert_allclose(out[0, 0, 0], 4.630845e-14, rtol=1e-5)

    @staticmethod
    def test_apply_psf(evaluator):
        flux = evaluator.compute_flux()
        npred = evaluator.apply_exposure(flux)
        out = evaluator.apply_psf(npred)
        assert out.data.shape == (3, 4, 5)
        assert_allclose(out.data.sum(), 2.2530737e-06, rtol=1e-5)
        assert_allclose(out.data[0, 0, 0], 2.407252e-08, rtol=1e-5)

    @staticmethod
    def test_apply_edisp(evaluator):
        flux = evaluator.compute_flux()
        npred = evaluator.apply_exposure(flux)
        out = evaluator.apply_edisp(npred)
        assert out.data.shape == (2, 4, 5)
        assert_allclose(out.data.sum(), 3.27582e-06, rtol=1e-5)
        assert_allclose(out.data[0, 0, 0], 4.630845e-08, rtol=1e-5)

    @staticmethod
    def test_compute_npred(evaluator):
        out = evaluator.compute_npred()
        assert out.data.shape == (2, 4, 5)
        assert_allclose(out.data.sum(), 2.253073467739508e-06, rtol=1e-5)
        assert_allclose(out.data[0, 0, 0], 2.407252e-08, rtol=1e-5)


def test_sky_point_source():
    # Test special case of point source. Regression test for GH 2367.

    energy_axis = MapAxis.from_edges([1, 10], unit="TeV", name="energy", interp="log")
    exposure = Map.create(
        skydir=(100, 70),
        npix=(4, 4),
        binsz=0.1,
        proj="AIT",
        unit="cm2 s",
        axes=[energy_axis],
    )
    exposure.data = np.ones_like(exposure.data)

    spatial_model = PointSpatialModel(
        lon_0=100.06 * u.deg, lat_0=70.03 * u.deg, frame="icrs"
    )
    # Create a spectral model with integral flux of 1 cm-2 s-1 in this energy band
    spectral_model = ConstantSpectralModel(const="1 cm-2 s-1 TeV-1")
    spectral_model.const.value /= spectral_model.integral(1 * u.TeV, 10 * u.TeV).value
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
    evaluator = MapEvaluator(model=model, exposure=exposure)
    flux = evaluator.compute_flux().to_value("cm-2 s-1")[0]

    expected = [
        [0, 0, 0, 0],
        [0, 0.140, 0.058, 0.0],
        [0, 0.564, 0.236, 0],
        [0, 0, 0, 0],
    ]
    assert_allclose(flux, expected, atol=0.01)

    assert_allclose(flux.sum(), 1)


@requires_data()
def test_fermi_isotropic():
    filename = "$GAMMAPY_DATA/fermi_3fhl/iso_P8R2_SOURCE_V6_v06.txt"
    model = create_fermi_isotropic_diffuse_model(filename)
    coords = {"lon": 0 * u.deg, "lat": 0 * u.deg, "energy": 50 * u.GeV}

    flux = model(**coords)

    assert_allclose(flux.value, 1.463e-13, rtol=1e-3)
    assert flux.unit == "MeV-1 cm-2 s-1 sr-1"

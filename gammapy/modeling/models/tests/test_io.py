# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import (
    MODEL_REGISTRY,
    AbsorbedSpectralModel,
    Absorption,
    BackgroundModel,
    Model,
    Models,
)
from gammapy.utils.scripts import read_yaml, write_yaml
from gammapy.utils.testing import requires_data


@requires_data()
def test_dict_to_skymodels():
    filename = get_pkg_data_filename("data/examples.yaml")
    models_data = read_yaml(filename)
    models = Models.from_dict(models_data)

    assert len(models) == 5

    model0 = models[0]
    assert isinstance(model0, BackgroundModel)
    assert model0.name == "background_irf"

    model0 = models[1]
    assert "ExpCutoffPowerLawSpectralModel" in model0.spectral_model.tag
    assert "PointSpatialModel" in model0.spatial_model.tag

    pars0 = model0.parameters
    assert pars0["index"].value == 2.1
    assert pars0["index"].unit == ""
    assert np.isnan(pars0["index"].max)
    assert np.isnan(pars0["index"].min)
    assert not pars0["index"].frozen

    assert pars0["lon_0"].value == -0.5
    assert pars0["lon_0"].unit == "deg"
    assert pars0["lon_0"].max == 180.0
    assert pars0["lon_0"].min == -180.0
    assert pars0["lon_0"].frozen

    assert pars0["lat_0"].value == -0.0005
    assert pars0["lat_0"].unit == "deg"
    assert pars0["lat_0"].max == 90.0
    assert pars0["lat_0"].min == -90.0
    assert pars0["lat_0"].frozen

    assert pars0["lambda_"].value == 0.006
    assert pars0["lambda_"].unit == "TeV-1"
    assert np.isnan(pars0["lambda_"].min)
    assert np.isnan(pars0["lambda_"].max)

    model1 = models[2]
    assert "pl" in model1.spectral_model.tag
    assert "PowerLawSpectralModel" in model1.spectral_model.tag
    assert "DiskSpatialModel" in model1.spatial_model.tag
    assert "disk" in model1.spatial_model.tag
    assert "LightCurveTemplateTemporalModel" in model1.temporal_model.tag

    pars1 = model1.parameters
    assert pars1["index"].value == 2.2
    assert pars1["index"].unit == ""
    assert pars1["lat_0"].scale == 1.0
    assert pars1["lat_0"].factor == pars1["lat_0"].value

    assert np.isnan(pars1["index"].max)
    assert np.isnan(pars1["index"].min)

    assert pars1["r_0"].unit == "deg"

    model2 = models[3]
    assert_allclose(model2.spectral_model.energy.data, [34.171, 44.333, 57.517])
    assert model2.spectral_model.energy.unit == "MeV"
    assert_allclose(
        model2.spectral_model.values.data, [2.52894e-06, 1.2486e-06, 6.14648e-06]
    )
    assert model2.spectral_model.values.unit == "1 / (cm2 MeV s)"

    assert "TemplateSpectralModel" in model2.spectral_model.tag
    assert "TemplateSpatialModel" in model2.spatial_model.tag

    assert not model2.spatial_model.normalize
    assert model2.spectral_model.parameters["norm"].value == 2.1

    # TODO problem of duplicate parameter name between TemplateSpatialModel and TemplateSpectralModel
    # assert model2.parameters["norm"].value == 2.1 # fail


@requires_data()
def test_sky_models_io(tmp_path):
    # TODO: maybe change to a test case where we create a model programatically?
    filename = get_pkg_data_filename("data/examples.yaml")
    models = Models.read(filename)
    models.covariance = np.eye(len(models.parameters))
    models.write(tmp_path / "tmp.yaml")
    models = Models.read(tmp_path / "tmp.yaml")
    assert models._covar_file == "tmp_covariance.dat"
    assert_allclose(models.covariance.data, np.eye(len(models.parameters)))
    assert_allclose(models.parameters["lat_0"].min, -90.0)

    # TODO: not sure if we should just round-trip, or if we should
    # check YAML file content (e.g. against a ref file in the repo)
    # or check serialised dict content


@requires_data()
def test_absorption_io(tmp_path):
    dominguez = Absorption.read_builtin("dominguez")
    model = AbsorbedSpectralModel(
        spectral_model=Model.create("pl", "spectral"),
        absorption=dominguez,
        redshift=0.5,
    )
    assert len(model.parameters) == 5

    model_dict = model.to_dict()
    parnames = [_["name"] for _ in model_dict["parameters"]]
    assert parnames == ["redshift", "alpha_norm"]

    new_model = AbsorbedSpectralModel.from_dict(model_dict)

    assert new_model.redshift.value == 0.5
    assert new_model.alpha_norm.name == "alpha_norm"
    assert new_model.alpha_norm.value == 1
    assert "PowerLawSpectralModel" in new_model.spectral_model.tag
    assert_allclose(new_model.absorption.energy, dominguez.energy)
    assert_allclose(new_model.absorption.param, dominguez.param)
    assert len(new_model.parameters) == 5

    test_absorption = Absorption(
        u.Quantity(range(3), "keV"),
        u.Quantity(range(2), ""),
        u.Quantity(np.ones((2, 3)), ""),
    )
    model = AbsorbedSpectralModel(
        spectral_model=Model.create("PowerLawSpectralModel", "spectral"),
        absorption=test_absorption,
        redshift=0.5,
    )
    model_dict = model.to_dict()
    new_model = AbsorbedSpectralModel.from_dict(model_dict)

    assert_allclose(new_model.absorption.energy, test_absorption.energy)
    assert_allclose(new_model.absorption.param, test_absorption.param)

    write_yaml(model_dict, tmp_path / "tmp.yaml")
    read_yaml(tmp_path / "tmp.yaml")


def make_all_models():
    """Make an instance of each model, for testing."""
    yield Model.create("ConstantSpatialModel", "spatial")
    map_constantmodel = Map.create(npix=(10, 20), unit="sr-1")
    yield Model.create("TemplateSpatialModel", "spatial", map=map_constantmodel)
    yield Model.create(
        "DiskSpatialModel", "spatial", lon_0="1 deg", lat_0="2 deg", r_0="3 deg"
    )
    yield Model.create("gauss", "spatial", lon_0="1 deg", lat_0="2 deg", sigma="3 deg")
    yield Model.create("PointSpatialModel", "spatial", lon_0="1 deg", lat_0="2 deg")
    yield Model.create(
        "ShellSpatialModel",
        "spatial",
        lon_0="1 deg",
        lat_0="2 deg",
        radius="3 deg",
        width="4 deg",
    )
    yield Model.create("ConstantSpectralModel", "spectral", const="99 cm-2 s-1 TeV-1")
    yield Model.create(
        "CompoundSpectralModel",
        "spectral",
        model1=Model.create("PowerLawSpectralModel", "spectral"),
        model2=Model.create("PowerLawSpectralModel", "spectral"),
        operator=np.add,
    )
    yield Model.create("PowerLawSpectralModel", "spectral")
    yield Model.create("PowerLawNormSpectralModel", "spectral")
    yield Model.create("PowerLaw2SpectralModel", "spectral")
    yield Model.create("ExpCutoffPowerLawSpectralModel", "spectral")
    yield Model.create("ExpCutoffPowerLawNormSpectralModel", "spectral")
    yield Model.create("ExpCutoffPowerLaw3FGLSpectralModel", "spectral")
    yield Model.create("SuperExpCutoffPowerLaw3FGLSpectralModel", "spectral")
    yield Model.create("SuperExpCutoffPowerLaw4FGLSpectralModel", "spectral")
    yield Model.create("LogParabolaSpectralModel", "spectral")
    yield Model.create("LogParabolaNormSpectralModel", "spectral")
    yield Model.create(
        "TemplateSpectralModel", "spectral", energy=[1, 2] * u.cm, values=[3, 4] * u.cm
    )  # TODO: add unit validation?
    yield Model.create("GaussianSpectralModel", "spectral")
    # TODO: yield Model.create("AbsorbedSpectralModel")
    # TODO: yield Model.create("NaimaSpectralModel")
    # TODO: yield Model.create("ScaleSpectralModel")
    yield Model.create("ConstantTemporalModel", "temporal")
    yield Model.create("LightCurveTemplateTemporalModel", "temporal", Table())
    yield Model.create(
        "SkyModel",
        spatial_model=Model.create("ConstantSpatialModel", "spatial"),
        spectral_model=Model.create("PowerLawSpectralModel", "spectral"),
    )
    m1 = Map.create(
        npix=(10, 20, 30), axes=[MapAxis.from_nodes([1, 2] * u.TeV, name="energy")]
    )
    yield Model.create("SkyDiffuseCube", map=m1)
    m2 = Map.create(
        npix=(10, 20, 30), axes=[MapAxis.from_edges([1, 2] * u.TeV, name="energy")]
    )
    yield Model.create("BackgroundModel", map=m2)


@pytest.mark.parametrize("model_class", MODEL_REGISTRY)
def test_all_model_classes(model_class):
    if isinstance(model_class.tag, list):
        assert model_class.tag[0] == model_class.__name__
    else:
        assert model_class.tag == model_class.__name__


@pytest.mark.parametrize("model", make_all_models())
def test_all_model_instances(model):
    tag = model.tag[0] if isinstance(model.tag, list) else model.tag
    assert tag == model.__class__.__name__


@requires_data()
def test_missing_parameters():
    filename = get_pkg_data_filename("data/examples.yaml")
    models = Models.read(filename)
    assert models["source1"].spatial_model.e in models.parameters
    assert len(models["source1"].spatial_model.parameters) == 6


def test_registries_print():
    print(MODEL_REGISTRY)

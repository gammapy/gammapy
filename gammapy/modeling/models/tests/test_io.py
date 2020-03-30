# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import (
    MODELS,
    AbsorbedSpectralModel,
    Absorption,
    Model,
    Models,
    BackgroundModel,
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
    assert model0.spectral_model.tag == "ExpCutoffPowerLawSpectralModel"
    assert model0.spatial_model.tag == "PointSpatialModel"

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
    assert model1.spectral_model.tag == "PowerLawSpectralModel"
    assert model1.spatial_model.tag == "DiskSpatialModel"
    assert model1.temporal_model.tag == "LightCurveTemplateTemporalModel"

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
    assert model2.spectral_model.values.unit == "1 / (cm2 MeV s sr)"

    assert model2.spectral_model.tag == "TemplateSpectralModel"
    assert model2.spatial_model.tag == "TemplateSpatialModel"

    assert model2.spatial_model.parameters["norm"].value == 1.0
    assert not model2.spatial_model.normalize
    assert model2.spectral_model.parameters["norm"].value == 2.1

    # TODO problem of duplicate parameter name between TemplateSpatialModel and TemplateSpectralModel
    # assert model2.parameters["norm"].value == 2.1 # fail


@requires_data()
def test_sky_models_io(tmp_path):
    # TODO: maybe change to a test case where we create a model programatically?
    filename = get_pkg_data_filename("data/examples.yaml")
    models = Models.read(filename)

    models.write(tmp_path / "tmp.yaml")
    models = Models.read(tmp_path / "tmp.yaml")

    assert_allclose(models.parameters["lat_0"].min, -90.0)

    # TODO: not sure if we should just round-trip, or if we should
    # check YAML file content (e.g. against a ref file in the repo)
    # or check serialised dict content


@requires_data()
def test_absorption_io(tmp_path):
    dominguez = Absorption.read_builtin("dominguez")
    model = AbsorbedSpectralModel(
        spectral_model=Model.create("PowerLawSpectralModel"),
        absorption=dominguez,
        redshift=0.5
    )
    assert len(model.parameters) == 5

    model_dict = model.to_dict()
    parnames = [_["name"] for _ in model_dict["parameters"]]
    assert parnames == ["redshift", "alpha_norm"]

    new_model = AbsorbedSpectralModel.from_dict(model_dict)

    assert new_model.redshift.value == 0.5
    assert new_model.alpha_norm.name == "alpha_norm"
    assert new_model.alpha_norm.value == 1
    assert new_model.spectral_model.tag == "PowerLawSpectralModel"
    assert_allclose(new_model.absorption.energy, dominguez.energy)
    assert_allclose(new_model.absorption.param, dominguez.param)
    assert len(new_model.parameters) == 5

    test_absorption = Absorption(
        u.Quantity(range(3), "keV"),
        u.Quantity(range(2), ""),
        u.Quantity(np.ones((2, 3)), ""),
    )
    model = AbsorbedSpectralModel(
        spectral_model=Model.create("PowerLawSpectralModel"),
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
    yield Model.create("ConstantSpatialModel")
    yield Model.create("TemplateSpatialModel", map=Map.create(npix=(10, 20)))
    yield Model.create("DiskSpatialModel", lon_0="1 deg", lat_0="2 deg", r_0="3 deg")
    yield Model.create(
        "GaussianSpatialModel", lon_0="1 deg", lat_0="2 deg", sigma="3 deg"
    )
    yield Model.create("PointSpatialModel", lon_0="1 deg", lat_0="2 deg")
    yield Model.create(
        "ShellSpatialModel", lon_0="1 deg", lat_0="2 deg", radius="3 deg", width="4 deg"
    )
    yield Model.create("ConstantSpectralModel", const="99 cm-2 s-1 TeV-1")
    # TODO: yield Model.create("CompoundSpectralModel")
    yield Model.create("PowerLawSpectralModel")
    yield Model.create("PowerLaw2SpectralModel")
    yield Model.create("ExpCutoffPowerLawSpectralModel")
    yield Model.create("ExpCutoffPowerLaw3FGLSpectralModel")
    yield Model.create("SuperExpCutoffPowerLaw3FGLSpectralModel")
    yield Model.create("SuperExpCutoffPowerLaw4FGLSpectralModel")
    yield Model.create("LogParabolaSpectralModel")
    yield Model.create(
        "TemplateSpectralModel", energy=[1, 2] * u.cm, values=[3, 4] * u.cm
    )  # TODO: add unit validation?
    yield Model.create("GaussianSpectralModel")
    # TODO: yield Model.create("AbsorbedSpectralModel")
    # TODO: yield Model.create("NaimaSpectralModel")
    # TODO: yield Model.create("ScaleSpectralModel")
    yield Model.create("ConstantTemporalModel")
    yield Model.create("LightCurveTemplateTemporalModel", Table())
    yield Model.create(
        "SkyModel",
        spatial_model=Model.create("ConstantSpatialModel"),
        spectral_model=Model.create("PowerLawSpectralModel"),
    )
    m1 = Map.create(
        npix=(10, 20, 30), axes=[MapAxis.from_nodes([1, 2] * u.TeV, name="energy")]
    )
    yield Model.create("SkyDiffuseCube", map=m1)
    m2 = Map.create(
        npix=(10, 20, 30), axes=[MapAxis.from_edges([1, 2] * u.TeV, name="energy")]
    )
    yield Model.create("BackgroundModel", map=m2)


@pytest.mark.parametrize("model_class", MODELS)
def test_all_model_classes(model_class):
    assert model_class.tag == model_class.__name__


@pytest.mark.parametrize("model", make_all_models())
def test_all_model_instances(model):
    assert model.tag == model.__class__.__name__


@requires_data()
def test_missing_parameters():
    filename = get_pkg_data_filename("data/examples.yaml")
    models = Models.read(filename)
    assert models["source1"].spatial_model.e in models.parameters
    assert len(models["source1"].spatial_model.parameters) == 6

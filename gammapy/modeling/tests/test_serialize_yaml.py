# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Datasets, Model
from gammapy.modeling.models import MODELS, AbsorbedSpectralModel, Absorption, SkyModels
from gammapy.modeling.serialize import dict_to_models
from gammapy.utils.scripts import read_yaml, write_yaml
from gammapy.utils.testing import requires_data


@requires_data()
def test_dict_to_skymodels():
    filename = get_pkg_data_filename("data/examples.yaml")
    models_data = read_yaml(filename)

    models = dict_to_models(models_data)

    assert len(models) == 3

    model0 = models[0]
    assert model0.spectral_model.tag == "ExpCutoffPowerLawSpectralModel"
    assert model0.spatial_model.tag == "PointSpatialModel"

    pars0 = model0.parameters
    assert pars0["index"].value == 2.1
    assert pars0["index"].unit == ""
    assert np.isnan(pars0["index"].max)
    assert np.isnan(pars0["index"].min)
    assert pars0["index"].frozen is False

    assert pars0["lon_0"].value == -50.0
    assert pars0["lon_0"].unit == "deg"
    assert pars0["lon_0"].max == 180.0
    assert pars0["lon_0"].min == -180.0
    assert pars0["lon_0"].frozen is True

    assert pars0["lat_0"].value == -0.05
    assert pars0["lat_0"].unit == "deg"
    assert pars0["lat_0"].max == 90.0
    assert pars0["lat_0"].min == -90.0
    assert pars0["lat_0"].frozen is True

    assert pars0["lambda_"].value == 0.06
    assert pars0["lambda_"].unit == "TeV-1"
    assert np.isnan(pars0["lambda_"].min)
    assert np.isnan(pars0["lambda_"].max)

    model1 = models[1]
    assert model1.spectral_model.tag == "PowerLawSpectralModel"
    assert model1.spatial_model.tag == "DiskSpatialModel"

    pars1 = model1.parameters
    assert pars1["index"].value == 2.2
    assert pars1["index"].unit == ""
    assert pars1["lat_0"].scale == 1.0
    assert pars1["lat_0"].factor == pars1["lat_0"].value

    assert np.isnan(pars1["index"].max)
    assert np.isnan(pars1["index"].min)

    assert pars1["r_0"].unit == "deg"

    model2 = models[2]
    assert_allclose(model2.spectral_model.energy.data, [34.171, 44.333, 57.517])
    assert model2.spectral_model.energy.unit == "MeV"
    assert_allclose(
        model2.spectral_model.values.data, [2.52894e-06, 1.2486e-06, 6.14648e-06]
    )
    assert model2.spectral_model.values.unit == "1 / (cm2 MeV s sr)"

    assert model2.spectral_model.tag == "TemplateSpectralModel"
    assert model2.spatial_model.tag == "TemplateSpatialModel"

    assert model2.spatial_model.parameters["norm"].value == 1.0
    assert model2.spatial_model.normalize is False
    assert model2.spectral_model.parameters["norm"].value == 2.1
    # TODO problem of duplicate parameter name between TemplateSpatialModel and TemplateSpectralModel
    # assert model2.parameters["norm"].value == 2.1 # fail


@requires_data()
def test_sky_models_io(tmp_path):
    # TODO: maybe change to a test case where we create a model programatically?
    filename = get_pkg_data_filename("data/examples.yaml")
    models = SkyModels.from_yaml(filename)

    models.to_yaml(tmp_path / "tmp.yaml")
    models = SkyModels.from_yaml(tmp_path / "tmp.yaml")

    assert_allclose(models.parameters["lat_0"].min, -90.0)

    # TODO: not sure if we should just round-trip, or if we should
    # check YAML file content (e.g. against a ref file in the repo)
    # or check serialised dict content


@requires_data()
def test_datasets_to_io(tmp_path):
    filedata = "$GAMMAPY_DATA/tests/models/gc_example_datasets.yaml"
    filemodel = "$GAMMAPY_DATA/tests/models/gc_example_models.yaml"

    datasets = Datasets.from_yaml(filedata, filemodel)

    assert len(datasets) == 2
    assert len(datasets.parameters) == 20

    dataset0 = datasets[0]
    assert dataset0.counts.data.sum() == 6824
    assert_allclose(dataset0.exposure.data.sum(), 2072125400000.0, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None

    assert_allclose(dataset0.background_model.evaluate().data.sum(), 4094.2, atol=0.1)

    assert dataset0.background_model.name == "background_irf_gc"

    dataset1 = datasets[1]
    assert dataset1.background_model.name == "background_irf_g09"

    assert dataset0.model["gll_iem_v06_cutout"] == dataset1.model["gll_iem_v06_cutout"]

    assert isinstance(dataset0.model, SkyModels)
    assert len(dataset0.model) == 2
    assert dataset0.model[0].name == "gc"
    assert dataset0.model[1].name == "gll_iem_v06_cutout"

    assert (
        dataset0.model[0].parameters["reference"]
        is dataset1.model[1].parameters["reference"]
    )

    assert_allclose(dataset1.model[1].parameters["lon_0"].value, 0.9, atol=0.1)

    datasets.to_yaml(tmp_path, prefix="written")
    datasets_read = Datasets.from_yaml(
        tmp_path / "written_datasets.yaml", tmp_path / "written_models.yaml"
    )
    assert len(datasets_read) == 2
    dataset0 = datasets_read[0]
    assert dataset0.counts.data.sum() == 6824
    assert_allclose(dataset0.exposure.data.sum(), 2072125400000.0, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None
    assert_allclose(dataset0.background_model.evaluate().data.sum(), 4094.2, atol=0.1)


@requires_data()
def test_absorption_io(tmp_path):
    dominguez = Absorption.read_builtin("dominguez")
    model = AbsorbedSpectralModel(
        spectral_model=Model.create("PowerLawSpectralModel"),
        absorption=dominguez,
        parameter=0.5,
        parameter_name="redshift",
    )
    assert len(model.parameters) == 5

    model_dict = model.to_dict()
    parnames = [_["name"] for _ in model_dict["parameters"]]
    assert parnames == ["redshift", "alpha_norm"]

    new_model = AbsorbedSpectralModel.from_dict(model_dict)

    assert new_model.parameter == 0.5
    assert new_model.parameter_name == "redshift"
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
        parameter=0.5,
        parameter_name="redshift",
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
    yield Model.create("LogGaussianSpectralModel")
    # TODO: yield Model.create("AbsorbedSpectralModel")
    # TODO: yield Model.create("NaimaSpectralModel")
    # TODO: yield Model.create("ScaleSpectralModel")
    yield Model.create("ConstantTemporalModel", norm=2)
    yield Model.create(
        "PhaseCurveTemplateTemporalModel", Table(), time_0=1, phase_0=2, f0=3
    )  # TODO: add table content validation?
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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import operator

import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from gammapy.maps import Map, MapAxis, RegionNDMap
from gammapy.modeling.models import (
    MODEL_REGISTRY,
    ConstantTemporalModel,
    EBLAbsorptionNormSpectralModel,
    ExpDecayTemporalModel,
    GaussianTemporalModel,
    LinearTemporalModel,
    Model,
    Models,
    PiecewiseNormSpectralModel,
    PowerLawSpectralModel,
    LogParabolaSpectralModel,
    PowerLawTemporalModel,
    CompoundSpectralModel,
    SineTemporalModel,
    SkyModel,
    TemplateNPredModel,
)
from gammapy.utils.scripts import read_yaml, write_yaml
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
@requires_data()
def models():
    filename = get_pkg_data_filename("./data/examples.yaml")
    models_data = read_yaml(filename)
    models = Models.from_dict(models_data)
    return models


@requires_data()
def test_dict_to_skymodels(models):

    assert len(models) == 5

    model0 = models[0]
    assert isinstance(model0, TemplateNPredModel)
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


@requires_data()
def test_sky_models_io(tmp_path, models):
    # TODO: maybe change to a test case where we create a model programmatically?
    models.covariance = np.eye(len(models.parameters))
    models.write(tmp_path / "tmp.yaml", full_output=True)
    models = Models.read(tmp_path / "tmp.yaml")
    assert models._covar_file == "tmp_covariance.dat"
    assert_allclose(models.covariance.data, np.eye(len(models.parameters)))
    assert_allclose(models.parameters["lat_0"].min, -90.0)

    # TODO: not sure if we should just round-trip, or if we should
    # check YAML file content (e.g. against a ref file in the repo)
    # or check serialised dict content


def test_piecewise_norm_spectral_model_init():
    with pytest.raises(ValueError):
        PiecewiseNormSpectralModel(
            energy=[
                1,
            ]
            * u.TeV,
            norms=[1, 5],
        )

    with pytest.raises(ValueError):
        PiecewiseNormSpectralModel(
            energy=[
                1,
            ]
            * u.TeV,
            norms=[
                1,
            ],
        )


def test_piecewise_norm_spectral_model_io():
    energy = [1, 3, 7, 10] * u.TeV
    norms = [1, 5, 3, 0.5] * u.Unit("")

    model = PiecewiseNormSpectralModel(energy=energy, norms=norms)
    model.parameters[0].value = 2

    model_dict = model.to_dict()

    parnames = [_["name"] for _ in model_dict["spectral"]["parameters"]]
    for k, parname in enumerate(parnames):
        assert parname == f"norm_{k}"

    new_model = PiecewiseNormSpectralModel.from_dict(model_dict)

    assert_allclose(new_model.parameters[0].value, 2)
    assert_allclose(new_model.energy, energy)
    assert_allclose(new_model.norms, [2, 5, 3, 0.5])


@requires_data()
def test_absorption_io(tmp_path):
    dominguez = EBLAbsorptionNormSpectralModel.read_builtin("dominguez", redshift=0.5)
    assert len(dominguez.parameters) == 2

    model_dict = dominguez.to_dict()
    parnames = [_["name"] for _ in model_dict["spectral"]["parameters"]]
    assert parnames == [
        "alpha_norm",
        "redshift",
    ]

    new_model = EBLAbsorptionNormSpectralModel.from_dict(model_dict)

    assert new_model.redshift.value == 0.5
    assert new_model.alpha_norm.name == "alpha_norm"
    assert new_model.alpha_norm.value == 1
    assert_allclose(new_model.energy, dominguez.energy)
    assert_allclose(new_model.param, dominguez.param)
    assert len(new_model.parameters) == 2

    model = EBLAbsorptionNormSpectralModel(
        u.Quantity(range(3), "keV"),
        u.Quantity(range(2), ""),
        u.Quantity(np.ones((2, 3)), ""),
        redshift=0.5,
        alpha_norm=1,
    )
    model_dict = model.to_dict()
    new_model = EBLAbsorptionNormSpectralModel.from_dict(model_dict)

    assert_allclose(new_model.energy, model.energy)
    assert_allclose(new_model.param, model.param)
    assert_allclose(new_model.data, model.data)

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
    yield Model.create("SuperExpCutoffPowerLaw4FGLDR3SpectralModel", "spectral")
    yield Model.create("SuperExpCutoffPowerLaw4FGLSpectralModel", "spectral")
    yield Model.create("LogParabolaSpectralModel", "spectral")
    yield Model.create("LogParabolaNormSpectralModel", "spectral")
    yield Model.create(
        "TemplateSpectralModel", "spectral", energy=[1, 2] * u.cm, values=[3, 4] * u.cm
    )  # TODO: add unit validation?
    yield Model.create(
        "PiecewiseNormSpectralModel",
        "spectral",
        energy=[1, 2] * u.cm,
        norms=[3, 4] * u.cm,
    )
    yield Model.create("GaussianSpectralModel", "spectral")
    # TODO: yield Model.create("EBLAbsorptionNormSpectralModel")
    # TODO: yield Model.create("NaimaSpectralModel")
    # TODO: yield Model.create("ScaleSpectralModel")
    yield Model.create("ConstantTemporalModel", "temporal")
    yield Model.create("LinearTemporalModel", "temporal")
    yield Model.create("PowerLawTemporalModel", "temporal")
    yield Model.create("SineTemporalModel", "temporal")
    m = RegionNDMap.create(region=None)
    yield Model.create("LightCurveTemplateTemporalModel", "temporal", m)
    yield Model.create(
        "SkyModel",
        spatial_model=Model.create("ConstantSpatialModel", "spatial"),
        spectral_model=Model.create("PowerLawSpectralModel", "spectral"),
    )
    m1 = Map.create(
        npix=(10, 20, 30), axes=[MapAxis.from_nodes([1, 2] * u.TeV, name="energy")]
    )
    yield Model.create("TemplateSpatialModel", "spatial", map=m1)
    m2 = Map.create(
        npix=(10, 20, 30), axes=[MapAxis.from_edges([1, 2] * u.TeV, name="energy")]
    )
    yield Model.create("TemplateNPredModel", map=m2)


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
def test_missing_parameters(models):
    assert models["source1"].spatial_model.e in models.parameters
    assert len(models["source1"].spatial_model.parameters) == 6


def test_simplified_output():
    model = PowerLawSpectralModel()
    full = model.to_dict(full_output=True)
    simplified = model.to_dict()
    for k, _ in enumerate(model.parameters.names):
        for item in ["min", "max", "error"]:
            assert item in full["spectral"]["parameters"][k]
            assert item not in simplified["spectral"]["parameters"][k]


def test_registries_print():
    assert "Registry" in str(MODEL_REGISTRY)


def test_io_temporal():
    classes = [
        ConstantTemporalModel,
        LinearTemporalModel,
        ExpDecayTemporalModel,
        GaussianTemporalModel,
        PowerLawTemporalModel,
        SineTemporalModel,
    ]
    for c in classes:
        sky_model = SkyModel(spectral_model=PowerLawSpectralModel(), temporal_model=c())
        model_dict = sky_model.to_dict()
        read_model = SkyModel.from_dict(model_dict)
        for p in sky_model.temporal_model.parameters:
            assert_allclose(read_model.temporal_model.parameters[p.name].value, p.value)
            assert read_model.temporal_model.parameters[p.name].unit == p.unit


@requires_data()
def test_link_label(models):
    skymodels = models.select(tag="sky-model")
    skymodels[0].spectral_model.reference = skymodels[1].spectral_model.reference
    dict_ = skymodels.to_dict()
    label0 = dict_["components"][0]["spectral"]["parameters"][2]["link"]
    label1 = dict_["components"][1]["spectral"]["parameters"][2]["link"]
    assert label0 == label1

    txt = skymodels.__str__()
    lines = txt.splitlines()
    n_link = 0
    for line in lines:
        if "@" in line:
            assert "reference" in line
            n_link += 1
    assert n_link == 2


def test_to_dict_not_default():

    model = PowerLawSpectralModel()
    model.index.min = -1
    model.index.max = -5
    model.index.frozen = True
    mdict = model.to_dict(full_output=False)

    index_dict = mdict["spectral"]["parameters"][0]
    assert "min" in index_dict
    assert "max" in index_dict
    assert "frozen" in index_dict
    assert "error" not in index_dict
    assert "interp" not in index_dict
    assert "scale_method" not in index_dict
    assert "is_norm" not in index_dict

    model_2 = model.from_dict(mdict)
    assert model_2.index.min == model.index.min
    assert model_2.index.max == model.index.max
    assert model_2.index.frozen == model.index.frozen


def test_compound_models_io(tmp_path):
    m1 = PowerLawSpectralModel()
    m2 = LogParabolaSpectralModel()
    m = CompoundSpectralModel(m1, m2, operator.add)
    sk = SkyModel(spectral_model=m, name="model")
    Models([sk]).write(tmp_path / "test.yaml")
    sk1 = Models.read(tmp_path / "test.yaml")
    assert_allclose(sk1.covariance.data, sk.covariance.data, rtol=1e-3)
    assert_allclose(np.sum(sk1.covariance.data), 0.0)
    assert Models([sk]).parameters_unique_names == [
        "model.spectral.index",
        "model.spectral.amplitude",
        "model.spectral.reference",
        "model.spectral.amplitude",
        "model.spectral.reference",
        "model.spectral.alpha",
        "model.spectral.beta",
    ]

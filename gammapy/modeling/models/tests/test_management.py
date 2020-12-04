import pytest
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from gammapy.modeling.models import (
    BackgroundModel,
    GaussianSpatialModel,
    Models,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.maps import Map, MapAxis, WcsGeom


@pytest.fixture(scope="session")
def backgrounds():
    axis = MapAxis.from_edges(np.logspace(-1, 1, 3), unit=u.TeV, name="energy")
    geom = WcsGeom.create(skydir=(0, 0), npix=(5, 4), frame="galactic", axes=[axis])
    m = Map.from_geom(geom)
    m.quantity = np.ones(geom.data_shape) * 1e-7
    background1 = BackgroundModel(m, name="bkg1", datasets_names="dataset-1")
    background2 = BackgroundModel(m, name="bkg2", datasets_names=["dataset-2"])
    backgrounds = [background1, background2]
    return backgrounds


@pytest.fixture(scope="session")
def models(backgrounds):
    spatial_model = GaussianSpatialModel(
        lon_0="3 deg", lat_0="4 deg", sigma="3 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model1 = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="source-1",
    )

    model2 = model1.copy(name="source-2")
    model2.datasets_names = ["dataset-1"]
    model3 = model1.copy(name="source-3")
    model3.datasets_names = "dataset-2"
    model3.spatial_model = PointSpatialModel()
    model3.parameters.freeze_all()
    models = Models([model1, model2, model3] + backgrounds)
    return models


def test_select(models):
    conditions = [
        {"datasets_names": "dataset-1"},
        {"datasets_names": "dataset-2"},
        {"datasets_names": ["dataset-1", "dataset-2"]},
        {"datasets_names": None},
        {"tag": "BackgroundModel"},
        {"tag": ["SkyModel", "BackgroundModel"]},
        {"tag": "point", "model_type": "spatial"},
        {"tag": ["point", "gauss"], "model_type": "spatial"},
        {"tag": "pl", "model_type": "spectral"},
        {"tag": ["pl", "pl-norm"], "model_type": "spectral"},
        {"name_substring": "bkg"},
        {"frozen": True},
        {"frozen": False},
        {"datasets_names": "dataset-1", "tag": "pl", "model_type": "spectral"},
    ]

    expected = [
        ["source-1", "source-2", "bkg1"],
        ["source-1", "source-3", "bkg2"],
        ["source-1", "source-2", "source-3", "bkg1", "bkg2"],
        ["source-1", "source-2", "source-3", "bkg1", "bkg2"],
        ["bkg1", "bkg2"],
        ["source-1", "source-2", "source-3", "bkg1", "bkg2"],
        ["source-3"],
        ["source-1", "source-2", "source-3"],
        ["source-1", "source-2", "source-3"],
        ["source-1", "source-2", "source-3", "bkg1", "bkg2"],
        ["bkg1", "bkg2"],
        ["source-3"],
        ["source-1", "source-2", "bkg1", "bkg2"],
        ["source-1", "source-2"],
    ]
    for cdt, xp in zip(conditions, expected):
        selected = models.select(**cdt)
        print(selected.names)
        assert selected.names == xp

    mask = models.mask(**conditions[4]) | models.mask(**conditions[6])
    selected = models[mask]
    assert selected.names == ["source-3", "bkg1", "bkg2"]


def test_restore_status(models):

    with models.parameters.restore_status(restore_values=True):
        models[1].spectral_model.amplitude.value = 0
        models[1].spectral_model.amplitude.frozen = True
        assert models[1].spectral_model.amplitude.value == 0
        assert models[1].spectral_model.amplitude.frozen == True
    assert_allclose(models[1].spectral_model.amplitude.value, 1e-11)
    assert models[1].spectral_model.amplitude.frozen == False

    with models.parameters.restore_status(restore_values=False):
        models[1].spectral_model.amplitude.value = 0
        models[1].spectral_model.amplitude.frozen = True
    assert models[1].spectral_model.amplitude.value == 0


def test_bounds(models):

    models.set_parameters_bounds(
        model_tag="pl", parameters_names="index", min=0, max=5, value=2.4
    )
    pl_mask = models.mask(spectral_tag="pl")
    assert np.all([m.spectral_model.index.value == 2.4 for m in models[pl_mask]])
    models.set_parameters_bounds(
        tag=["pl", "pl-norm"],
        model_type="spectral",
        parameters_names=["norm", "amplitude"],
        min=0,
        max=None,
    )
    bkg_mask = models.mask(tag="BackgroundModel")
    assert np.all([m.spectral_model.amplitude.min == 0 for m in models[pl_mask]])
    assert np.all([m._spectral_model.norm.min == 0 for m in models[bkg_mask]])


def test_freeze(models):

    models.freeze()
    assert np.all([p.frozen for p in models.parameters])
    models.unfreeze()
    assert models["source-1"].spatial_model.lon_0.frozen == False
    assert models["source-1"].spectral_model.reference.frozen == True
    assert models["source-3"].spatial_model.lon_0.frozen == False
    assert models["bkg1"]._spectral_model.tilt.frozen == True
    assert models["bkg1"]._spectral_model.norm.frozen == False
    models.freeze("spatial")
    assert models["source-1"].spatial_model.lon_0.frozen == True
    assert models["source-3"].spatial_model.lon_0.frozen == True
    assert models["bkg1"]._spectral_model.norm.frozen == False
    models.unfreeze("spatial")
    assert models["source-1"].spatial_model.lon_0.frozen == False
    assert models["source-1"].spectral_model.reference.frozen == True
    assert models["source-3"].spatial_model.lon_0.frozen == False
    models.freeze("spectral")
    assert models["bkg1"]._spectral_model.tilt.frozen == True
    assert models["bkg1"]._spectral_model.norm.frozen == True
    assert models["source-1"].spectral_model.index.frozen == True
    assert models["source-3"].spatial_model.lon_0.frozen == False
    models.unfreeze("spectral")
    assert models["bkg1"]._spectral_model.tilt.frozen == True
    assert models["bkg1"]._spectral_model.norm.frozen == False
    assert models["source-1"].spectral_model.index.frozen == False

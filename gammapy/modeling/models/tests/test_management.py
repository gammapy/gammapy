import pytest
import numpy as np
import astropy.units as u
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
        {"spatial_tag": "point"},
        {"spatial_tag": ["point", "gauss"]},
        {"spectral_tag": "pl"},
        {"spectral_tag": ["pl", "pl-norm"]},
        {"name_substring": "bkg"},
        {"frozen": True},
        {"frozen": False},
        {"datasets_names": "dataset-1", "spectral_tag": "pl"},
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
        mask = models.mask(**cdt)
        selected = models.select(mask)
        assert selected.names == xp

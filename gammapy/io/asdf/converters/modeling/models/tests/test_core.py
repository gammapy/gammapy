# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import operator
import pytest

from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import (
    CompoundSpectralModel,
    FoVBackgroundModel,
    GaussianSpatialModel,
    GaussianTemporalModel,
    LogParabolaSpectralModel,
    Model,
    Models,
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
)
from numpy.testing import assert_allclose

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")


def test_models_asdf_roundtrip_skymodel(tmp_path):
    file_path = tmp_path / "test.asdf"
    spatial_model = GaussianSpatialModel(
        lon_0="10 deg", lat_0="5 deg", sigma="0.5 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel()
    spectral_model.index.frozen = True
    sky_model = SkyModel(
        spectral_model=spectral_model,
        spatial_model=spatial_model,
        name="SkyModel",
    )
    models = Models([sky_model])
    with asdf.AsdfFile() as af:
        af["models"] = models
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["models"]
    for actual, desired in zip(models[0].parameters, result[0].parameters):
        assert_allclose(actual.value, desired.value)
        assert actual.unit == desired.unit
        assert actual.name == desired.name
        assert actual.frozen == desired.frozen
    assert result[0].spatial_model.frame == "galactic"


def test_models_asdf_roundtrip_temporal(tmp_path):
    file_path = tmp_path / "test.asdf"
    temporal_model = GaussianTemporalModel(t_ref=50003.2503033 * u.d, sigma="2.43 day")
    sky_model = SkyModel(
        spectral_model=PowerLawSpectralModel(),
        temporal_model=temporal_model,
        name="test-temporal-model",
    )
    models = Models([sky_model])
    with asdf.AsdfFile() as af:
        af["models"] = models
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["models"]
    for actual, desired in zip(models[0].parameters, result[0].parameters):
        assert_allclose(actual.value, desired.value)
        assert actual.unit == desired.unit
        assert actual.name == desired.name
        assert actual.frozen == desired.frozen


def test_models_asdf_roundtrip_compound_spectral(tmp_path):
    file_path = tmp_path / "test.asdf"
    model1 = PowerLawSpectralModel()
    model2 = LogParabolaSpectralModel()
    spectral_model = CompoundSpectralModel(model1, model2, operator.add)
    sky_model = SkyModel(spectral_model=spectral_model, name="compound-source")
    models = Models([sky_model])
    with asdf.AsdfFile() as af:
        af["models"] = models
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["models"]
    for actual, desired in zip(models[0].parameters, result[0].parameters):
        assert_allclose(actual.value, desired.value)
        assert actual.unit == desired.unit
        assert actual.name == desired.name
        assert actual.frozen == desired.frozen

    assert result[0].spectral_model.operator == operator.add


def test_models_asdf_roundtrip_fovbackground(tmp_path):
    file_path = tmp_path / "test.asdf"
    bkg_model = FoVBackgroundModel(dataset_name="test-models-asdf")
    models = Models([bkg_model])
    with asdf.AsdfFile() as af:
        af["models"] = models
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["models"]
    assert result[0].datasets_names == bkg_model.datasets_names


def test_models_asdf_roundtrip_template_npred(tmp_path):
    file_path = tmp_path / "test.asdf"
    filename = str(tmp_path / "template_npred.fits")
    m = Map.create(
        npix=(10, 20, 30), axes=[MapAxis.from_edges([1, 2] * u.TeV, name="energy")]
    )
    m.data = np.arange(m.data.size, dtype=m.data.dtype).reshape(m.data.shape)
    template_model = Model.create("TemplateNPredModel", map=m, filename=filename)

    models = Models([template_model])
    with asdf.AsdfFile() as af:
        af["models"] = models
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["models"]
    for actual, desired in zip(models[0].parameters, result[0].parameters):
        assert_allclose(actual.value, desired.value)
        assert actual.unit == desired.unit
        assert actual.name == desired.name
        assert actual.frozen == desired.frozen

    assert_allclose(result[0].map.data, m.data)


def test_models_asdf_roundtrip_template_spatial(tmp_path):
    file_path = tmp_path / "test.asdf"
    filename = str(tmp_path / "template_spatial.fits")
    m = Map.create(npix=(10, 20))
    m.data = np.arange(m.data.size, dtype=m.data.dtype).reshape(m.data.shape)
    template_model = TemplateSpatialModel(m, filename=filename, normalize=False)

    sky_model = SkyModel(
        spectral_model=PowerLawSpectralModel(),
        spatial_model=template_model,
        name="template-spatial",
    )
    models = Models([sky_model])
    with asdf.AsdfFile() as af:
        af["models"] = models
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["models"]
    for actual, desired in zip(models[0].parameters, result[0].parameters):
        assert_allclose(actual.value, desired.value)
        assert actual.unit == desired.unit
        assert actual.name == desired.name
        assert actual.frozen == desired.frozen

    assert_allclose(result[0].spatial_model.map.data, template_model.map.data)


def test_models_asdf_roundtrip_covariance(tmp_path):
    file_path = tmp_path / "test.asdf"
    model = SkyModel(spectral_model=PowerLawSpectralModel(), name="test-models-asdf")
    models = Models([model])
    n = len(models.parameters)
    cov = np.eye(n)
    if n > 1:
        cov[0, 1] = cov[1, 0] = 0.3
    models.covariance = cov

    with asdf.AsdfFile() as af:
        af["models"] = models
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["models"]

    assert_allclose(result.covariance.data, models.covariance.data)


def test_models_asdf_roundtrip_linked_parameters(tmp_path):
    file_path = tmp_path / "test.asdf"
    model1 = SkyModel(
        spectral_model=PowerLawSpectralModel(
            index=2.3, amplitude="4 cm-2 s-1 TeV-1", reference="2 TeV"
        ),
        name="model-1",
    )
    model2 = SkyModel(spectral_model=PowerLawSpectralModel(), name="model-2")

    model2.spectral_model.index = model1.spectral_model.index

    models = Models([model1, model2])

    with asdf.AsdfFile() as af:
        af["models"] = models
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["models"]

    assert result[0].spectral_model.index is result[1].spectral_model.index

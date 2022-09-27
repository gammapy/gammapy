# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import MapDataset, SpectrumDataset
from gammapy.makers import FoVBackgroundMaker, MapDatasetMaker, SafeMaskMaker
from gammapy.maps import Map, MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    FoVBackgroundModel,
    PointSpatialModel,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def observation():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_id = 23523
    return datastore.obs(obs_id)


@pytest.fixture(scope="session")
def geom():
    energy_axis = MapAxis.from_edges([1, 10], unit="TeV", name="energy", interp="log")
    return WcsGeom.create(
        skydir=SkyCoord(83.633, 22.014, unit="deg"),
        binsz=0.02,
        width=(5, 5),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis],
    )


@pytest.fixture(scope="session")
def reference(geom):
    return MapDataset.create(geom)


@pytest.fixture(scope="session")
def exclusion_mask(geom):
    """Example mask for testing."""
    pos = SkyCoord(83.633, 22.014, unit="deg", frame="icrs")
    region = CircleSkyRegion(pos, Angle(0.3, "deg"))
    return ~geom.region_mask([region])


@pytest.fixture(scope="session")
def obs_dataset(geom, observation):
    safe_mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max="2 deg")
    map_dataset_maker = MapDatasetMaker(selection=["counts", "background", "exposure"])

    reference = MapDataset.create(geom)
    cutout = reference.cutout(
        observation.pointing_radec, width="4 deg", name="test-fov"
    )

    dataset = map_dataset_maker.run(cutout, observation)
    dataset = safe_mask_maker.run(dataset, observation)
    return dataset


def test_fov_bkg_maker_incorrect_method():
    with pytest.raises(ValueError):
        FoVBackgroundMaker(method="bad")


@requires_data()
def test_fov_bkg_maker_scale(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)
    test_dataset = obs_dataset.copy(name="test-fov")

    dataset = fov_bkg_maker.run(test_dataset)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 0.83, rtol=1e-2)
    assert_allclose(model.norm.error, 0.0207, rtol=1e-2)
    assert_allclose(model.tilt.value, 0.0, rtol=1e-2)


@requires_data()
def test_fov_bkg_maker_scale_nocounts(obs_dataset, exclusion_mask, caplog):
    fov_bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)
    test_dataset = obs_dataset.copy(name="test-fov")
    test_dataset.counts *= 0

    dataset = fov_bkg_maker.run(test_dataset)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 1, rtol=1e-4)
    assert_allclose(model.tilt.value, 0.0, rtol=1e-2)
    assert "WARNING" in [_.levelname for _ in caplog.records]
    message1 = (
        "FoVBackgroundMaker failed. Only 0 counts outside exclusion mask "
        "for test-fov. Setting mask to False."
    )
    assert message1 in [_.message for _ in caplog.records]


@requires_data()
def test_fov_bkg_maker_fit(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    dataset1 = obs_dataset.copy(name="test-fov")
    dataset = fov_bkg_maker.run(dataset1)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 0.83077, rtol=1e-4)
    assert_allclose(model.norm.error, 0.02069, rtol=1e-2)
    assert_allclose(model.tilt.value, 0.0, rtol=1e-4)
    assert_allclose(model.tilt.error, 0.0, rtol=1e-2)
    assert_allclose(fov_bkg_maker.default_spectral_model.tilt.value, 0.0)
    assert_allclose(fov_bkg_maker.default_spectral_model.norm.value, 1.0)

    spectral_model = PowerLawNormSpectralModel()
    spectral_model.tilt.frozen = False
    fov_bkg_maker = FoVBackgroundMaker(
        method="fit", exclusion_mask=exclusion_mask, spectral_model=spectral_model
    )

    dataset2 = obs_dataset.copy(name="test-fov")
    dataset = fov_bkg_maker.run(dataset2)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 0.901523, rtol=1e-4)
    assert_allclose(model.tilt.value, 0.071069, rtol=1e-4)

    # TODO: reactivate with a more stable error estimate
    # assert_allclose(model.norm.error, 0.355637, rtol=1e-2)
    # assert_allclose(model.tilt.error, 0.342201, rtol=1e-2)

    assert_allclose(fov_bkg_maker.default_spectral_model.tilt.value, 0.0)
    assert_allclose(fov_bkg_maker.default_spectral_model.norm.value, 1.0)


@requires_data()
def test_fov_bkg_maker_fit_nocounts(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy(name="test-fov")
    test_dataset.counts.data[...] = 0

    dataset = fov_bkg_maker.run(test_dataset)
    assert np.all(dataset.mask_safe.data == 0)


@requires_data()
def test_fov_bkg_maker_with_source_model(obs_dataset, exclusion_mask, caplog):

    test_dataset = obs_dataset.copy(name="test-fov")

    # crab model
    spatial_model = PointSpatialModel(
        lon_0="83.619deg", lat_0="22.024deg", frame="icrs"
    )
    spectral_model = PowerLawSpectralModel(
        index=2.6, amplitude="4.5906e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="test-source"
    )

    bkg_model = FoVBackgroundModel(dataset_name="test-fov")
    test_dataset.models = [model, bkg_model]

    # pre-fit both source and background to get reference model
    Fit().run(test_dataset)
    bkg_model_spec = test_dataset.models[f"{test_dataset.name}-bkg"].spectral_model
    norm_ref = 0.897
    assert not bkg_model_spec.norm.frozen
    assert_allclose(bkg_model_spec.norm.value, norm_ref, rtol=1e-4)
    assert_allclose(bkg_model_spec.tilt.value, 0.0, rtol=1e-4)

    # apply scale method with pre-fitted source model and no exclusion_mask
    bkg_model_spec.norm.value = 1
    fov_bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=None)
    dataset = fov_bkg_maker.run(test_dataset)

    bkg_model_spec = test_dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(bkg_model_spec.norm.value, norm_ref, rtol=1e-4)
    assert_allclose(bkg_model_spec.tilt.value, 0.0, rtol=1e-4)

    # apply fit method with pre-fitted source model and no exclusion mask
    bkg_model_spec.norm.value = 1
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=None)
    dataset = fov_bkg_maker.run(test_dataset)

    bkg_model_spec = test_dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(bkg_model_spec.norm.value, norm_ref, rtol=1e-4)
    assert_allclose(bkg_model_spec.tilt.value, 0.0, rtol=1e-4)

    # apply scale method with pre-fitted source model and exclusion_mask
    bkg_model_spec.norm.value = 1
    fov_bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)
    dataset = fov_bkg_maker.run(test_dataset)

    bkg_model_spec = test_dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(bkg_model_spec.norm.value, 0.830779, rtol=1e-4)
    assert_allclose(bkg_model_spec.tilt.value, 0.0, rtol=1e-4)

    # apply fit method with pre-fitted source model and exclusion mask
    bkg_model_spec.norm.value = 1
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)
    dataset = fov_bkg_maker.run(test_dataset)

    bkg_model_spec = test_dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(bkg_model_spec.norm.value, 0.830779, rtol=1e-4)
    assert_allclose(bkg_model_spec.tilt.value, 0.0, rtol=1e-4)

    # Here we check that source parameters are correctly thawed after fit.
    assert not dataset.models.parameters["index"].frozen
    assert not dataset.models.parameters["lon_0"].frozen

    # test
    model.spectral_model.amplitude.value *= 1e5
    fov_bkg_maker = FoVBackgroundMaker(method="scale")
    dataset = fov_bkg_maker.run(test_dataset)
    assert "WARNING" in [_.levelname for _ in caplog.records]
    message1 = (
        "FoVBackgroundMaker failed. Negative residuals counts for"
        " test-fov. Setting mask to False."
    )
    assert message1 in [_.message for _ in caplog.records]


@requires_data()
def test_fov_bkg_maker_fit_with_tilt(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(
        method="fit",
        exclusion_mask=exclusion_mask,
    )

    test_dataset = obs_dataset.copy(name="test-fov")

    model = FoVBackgroundModel(dataset_name="test-fov")
    model.spectral_model.tilt.frozen = False
    test_dataset.models = [model]
    dataset = fov_bkg_maker.run(test_dataset)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 0.901523, rtol=1e-4)
    assert_allclose(model.tilt.value, 0.071069, rtol=1e-4)


@requires_data()
def test_fov_bkg_maker_fit_fail(obs_dataset, exclusion_mask, caplog):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy(name="test-fov")

    # Putting null background model to prevent convergence
    test_dataset.background.data *= 0
    dataset = fov_bkg_maker.run(test_dataset)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 1, rtol=1e-4)
    assert "WARNING" in [_.levelname for _ in caplog.records]
    message1 = (
        "FoVBackgroundMaker failed. Non-finite normalisation value for "
        "test-fov. Setting mask to False."
    )
    assert message1 in [_.message for _ in caplog.records]


@requires_data()
def test_fov_bkg_maker_scale_fail(obs_dataset, exclusion_mask, caplog):
    fov_bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy(name="test-fov")
    # Putting negative background model to prevent correct scaling
    test_dataset.background.data *= -1
    dataset = fov_bkg_maker.run(test_dataset)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 1, rtol=1e-4)
    assert "WARNING" in [_.levelname for _ in caplog.records]
    message1 = (
        "FoVBackgroundMaker failed. Only -1940 background counts outside"
        " exclusion mask for test-fov. Setting mask to False."
    )
    assert message1 in [_.message for _ in caplog.records]


@requires_data()
def test_fov_bkg_maker_mask_fit_handling(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)
    test_dataset = obs_dataset.copy(name="test-fov")
    region = CircleSkyRegion(obs_dataset._geom.center_skydir, Angle(0.4, "deg"))
    mask_fit = obs_dataset._geom.region_mask(regions=[region])
    test_dataset.mask_fit = mask_fit

    dataset = fov_bkg_maker.run(test_dataset)
    assert np.all(test_dataset.mask_fit == mask_fit)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 0.9975, rtol=1e-3)
    assert_allclose(model.norm.error, 0.1115, rtol=1e-3)
    assert_allclose(model.tilt.value, 0.0, rtol=1e-2)


@requires_data()
def test_fov_bkg_maker_spectrumdataset(obs_dataset):
    from regions import CircleSkyRegion

    maker = FoVBackgroundMaker()
    energy_axis = MapAxis.from_edges([1, 10], unit="TeV", name="energy", interp="log")
    region = CircleSkyRegion(obs_dataset._geom.center_skydir, Angle("0.1 deg"))
    geom = RegionGeom.create(region, axes=[energy_axis])
    dataset = SpectrumDataset.create(geom)

    with pytest.raises(TypeError):
        maker.run(dataset)

    region_dataset = obs_dataset.to_region_map_dataset(region)
    with pytest.raises(TypeError):
        maker.run(region_dataset)


def test_fov_background_maker_str():
    exclusion_mask = Map.create(binsz=0.2, width=(2, 2))
    maker_fov = FoVBackgroundMaker(exclusion_mask=exclusion_mask)
    assert "FoVBackgroundMaker" in str(maker_fov)

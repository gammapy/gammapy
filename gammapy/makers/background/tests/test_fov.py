# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import MapDataset
from gammapy.makers import FoVBackgroundMaker, MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    FoVBackgroundModel,
    GaussianSpatialModel,
    PowerLawSpectralModel,
    PowerLawNormSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data, requires_dependency


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
def test_fov_bkg_maker_scale(obs_dataset, exclusion_mask, caplog):
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
    assert caplog.records[-1].levelname == "WARNING"
    assert (
        "Only 0 counts outside exclusion mask for test-fov"
        in caplog.records[-1].message
    )
    assert "FoVBackgroundMaker failed" in caplog.records[-1].message


@requires_data()
@requires_dependency("iminuit")
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
    assert_allclose(model.norm.error, 0.583411, rtol=1e-2)
    assert_allclose(model.tilt.value, 0.071069, rtol=1e-4)
    assert_allclose(model.tilt.error, 0.562129, rtol=1e-2)
    assert_allclose(fov_bkg_maker.default_spectral_model.tilt.value, 0.0)
    assert_allclose(fov_bkg_maker.default_spectral_model.norm.value, 1.0)


@pytest.mark.xfail
@requires_data()
@requires_dependency("iminuit")
def test_fov_bkg_maker_fit_nocounts(obs_dataset, exclusion_mask, caplog):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy(name="test-fov")
    test_dataset.counts *= 0

    dataset = fov_bkg_maker.run(test_dataset)

    # This should be solved along with issue https://github.com/gammapy/gammapy/issues/3175
    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 1, rtol=1e-4)
    assert_allclose(model.tilt.value, 0.0, rtol=1e-4)

    assert caplog.records[-1].levelname == "WARNING"
    assert f"Fit did not converge for {dataset.name}" in caplog.records[-1].message


@requires_data()
@requires_dependency("iminuit")
def test_fov_bkg_maker_fit_with_source_model(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy(name="test-fov")
    spatial_model = GaussianSpatialModel(
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.2 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="test-source"
    )

    bkg_model = FoVBackgroundModel(dataset_name="test-fov")
    test_dataset.models = [model, bkg_model]

    dataset = fov_bkg_maker.run(test_dataset)

    # Here we check that source parameters are correctly thawed after fit.
    assert not dataset.models.parameters["index"].frozen
    assert not dataset.models.parameters["lon_0"].frozen

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert not model.norm.frozen
    assert_allclose(model.norm.value, 0.830789, rtol=1e-4)
    assert_allclose(model.tilt.value, 0.0, rtol=1e-4)


@requires_data()
@requires_dependency("iminuit")
def test_fov_bkg_maker_fit_with_tilt(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask,)

    test_dataset = obs_dataset.copy(name="test-fov")

    model = FoVBackgroundModel(dataset_name="test-fov")
    model.spectral_model.tilt.frozen = False
    test_dataset.models = [model]
    dataset = fov_bkg_maker.run(test_dataset)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 0.901523, rtol=1e-4)
    assert_allclose(model.tilt.value, 0.071069, rtol=1e-4)


@requires_data()
@requires_dependency("iminuit")
def test_fov_bkg_maker_fit_fail(obs_dataset, exclusion_mask, caplog):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy(name="test-fov")

    # Putting null background model to prevent convergence
    test_dataset.background.data *= 0
    dataset = fov_bkg_maker.run(test_dataset)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 1, rtol=1e-4)
    assert caplog.records[-1].levelname == "WARNING"
    assert f"Fit did not converge for {dataset.name}" in caplog.records[-1].message


@requires_data()
def test_fov_bkg_maker_scale_fail(obs_dataset, exclusion_mask, caplog):
    fov_bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy(name="test-fov")
    # Putting negative background model to prevent correct scaling
    test_dataset.background.data *= -1
    dataset = fov_bkg_maker.run(test_dataset)

    model = dataset.models[f"{dataset.name}-bkg"].spectral_model
    assert_allclose(model.norm.value, 1, rtol=1e-4)
    assert caplog.records[-1].levelname == "WARNING"
    assert (
        f"Only -1940 background counts outside exclusion mask for test-fov"
        in caplog.records[-1].message
    )
    assert "FoVBackgroundMaker failed" in caplog.records[-1].message

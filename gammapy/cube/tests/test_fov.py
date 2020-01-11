# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.datasets import MapDataset
from gammapy.cube.fov import FoVBackgroundMaker
from gammapy.cube.make import MapDatasetMaker, SafeMaskMaker
from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap
from gammapy.utils.testing import requires_data, requires_dependency
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)


@pytest.fixture(scope="session")
def observation():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_id = 23523
    return datastore.obs(obs_id)


@pytest.fixture(scope="session")
def geom():
    energy_axis = MapAxis.from_edges([1, 10], unit="TeV", name="ENERGY", interp="log")
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
    exclusion = WcsNDMap.from_geom(geom)
    exclusion.data = geom.region_mask([region], inside=False)
    return exclusion


@pytest.fixture(scope="session")
def obs_dataset(geom, observation):
    safe_mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max="2 deg")
    map_dataset_maker = MapDatasetMaker(selection=["counts", "background", "exposure"])

    reference = MapDataset.create(geom)
    cutout = reference.cutout(observation.pointing_radec, width="4 deg")

    dataset = map_dataset_maker.run(cutout, observation)
    dataset = safe_mask_maker.run(dataset, observation)
    return dataset


def test_fov_bkg_maker_incorrect_method():
    with pytest.raises(ValueError):
        FoVBackgroundMaker(method="bad")


@requires_data()
def test_fov_bkg_maker_scale(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy()
    dataset = fov_bkg_maker.run(test_dataset)

    assert_allclose(dataset.background_model.norm.value, 0.83078, rtol=1e-4)
    assert_allclose(dataset.background_model.tilt.value, 0.0, rtol=1e-4)


@requires_data()
@requires_dependency("iminuit")
def test_fov_bkg_maker_fit(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy()
    dataset = fov_bkg_maker.run(test_dataset)

    assert_allclose(dataset.background_model.norm.value, 0.8307, rtol=1e-4)
    assert_allclose(dataset.background_model.tilt.value, 0.0, rtol=1e-4)


@requires_data()
@requires_dependency("iminuit")
def test_fov_bkg_maker_fit_with_source_model(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy()
    spatial_model = GaussianSpatialModel(
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.2 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
    test_dataset.models = model
    dataset = fov_bkg_maker.run(test_dataset)

    # Here we check that source parameters are correctly thawed after fit.
    assert dataset.models.parameters["index"].frozen is False
    assert dataset.models.parameters["lon_0"].frozen is False
    assert dataset.background_model.norm.frozen is False

    assert_allclose(dataset.background_model.norm.value, 0.8307, rtol=1e-4)
    assert_allclose(dataset.background_model.tilt.value, 0.0, rtol=1e-4)


@requires_data()
@requires_dependency("iminuit")
def test_fov_bkg_maker_fit_with_tilt(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy()
    test_dataset.background_model.tilt.frozen = False
    dataset = fov_bkg_maker.run(test_dataset)

    assert_allclose(dataset.background_model.norm.value, 0.9034, rtol=1e-4)
    assert_allclose(dataset.background_model.tilt.value, 0.0728, rtol=1e-4)


@requires_data()
@requires_dependency("iminuit")
def test_fov_bkg_maker_fit_fail(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy()
    # Putting negative background model to prevent convergence
    test_dataset.background_model.map.data *= -1
    dataset = fov_bkg_maker.run(test_dataset)

    assert_allclose(dataset.background_model.norm.value, 1, rtol=1e-4)

@requires_data()
def test_fov_bkg_maker_scale_fail(obs_dataset, exclusion_mask):
    fov_bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)

    test_dataset = obs_dataset.copy()
    # Putting negative background model to prevent correct scaling
    test_dataset.background_model.map.data *= -1
    dataset = fov_bkg_maker.run(test_dataset)

    assert_allclose(dataset.background_model.norm.value, 1, rtol=1e-4)

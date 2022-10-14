# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import (
    CircleSkyRegion,
    EllipseAnnulusSkyRegion,
    EllipseSkyRegion,
    PointSkyRegion,
    RectangleSkyRegion,
)
from gammapy.data import DataStore
from gammapy.datasets import SpectrumDataset
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    ReflectedRegionsFinder,
    SafeMaskMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
)
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.utils.regions import compound_region_to_regions
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@pytest.fixture(scope="session")
def exclusion_mask():
    """Example mask for testing."""
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    exclusion_region = CircleSkyRegion(pos, Angle(0.3, "deg"))
    geom = WcsGeom.create(skydir=pos, binsz=0.02, width=10.0)
    return ~geom.region_mask([exclusion_region])


@pytest.fixture(scope="session")
def on_region():
    """Example on_region for testing."""
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)
    return region


@pytest.fixture(scope="session")
def observations():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs_ids = [23523, 23526]
    return datastore.get_observations(obs_ids)


@pytest.fixture(scope="session")
def observations_fixed_rad_max():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/joint-crab/dl3/magic/")
    obs_ids = [5029748]
    return datastore.get_observations(obs_ids, required_irf="point-like")


@pytest.fixture()
def reflected_bkg_maker(exclusion_mask):
    finder = ReflectedRegionsFinder()
    return ReflectedRegionsBackgroundMaker(
        region_finder=finder,
        exclusion_mask=exclusion_mask,
    )


region_finder_param = [
    (SkyCoord(83.2, 22.5, unit="deg"), 15, Angle("82.592 deg"), 17, 17),
    (SkyCoord(84.2, 22.5, unit="deg"), 17, Angle("83.636 deg"), 19, 19),
    (SkyCoord(83.2, 21.5, unit="deg"), 15, Angle("83.672 deg"), 17, 17),
]


@requires_data()
@pytest.mark.parametrize(
    "pointing_pos, nreg1, reg3_ra, nreg2, nreg3", region_finder_param
)
def test_find_reflected_regions(
    exclusion_mask, on_region, pointing_pos, nreg1, reg3_ra, nreg2, nreg3
):
    pointing = pointing_pos
    finder = ReflectedRegionsFinder(
        min_distance_input="0 deg",
    )
    regions, _ = finder.run(
        center=pointing,
        region=on_region,
        exclusion_mask=exclusion_mask,
    )
    assert len(regions) == nreg1
    assert_quantity_allclose(regions[3].center.icrs.ra, reg3_ra, rtol=1e-2)

    # Test without exclusion
    regions, _ = finder.run(center=pointing, region=on_region)
    assert len(regions) == nreg2

    # Test with too small exclusion
    small_mask = exclusion_mask.cutout(pointing, Angle("0.1 deg"))
    regions, _ = finder.run(
        center=pointing,
        region=on_region,
        exclusion_mask=small_mask,
    )
    assert len(regions) == nreg3

    # Test with maximum number of regions
    finder.max_region_number = 5
    regions, _ = finder.run(
        center=pointing,
        region=on_region,
        exclusion_mask=small_mask,
    )
    assert len(regions) == 5

    # Test with an other type of region
    on_ellipse_annulus = EllipseAnnulusSkyRegion(
        center=on_region.center.galactic,
        inner_width=0.1 * u.deg,
        outer_width=0.2 * u.deg,
        inner_height=0.3 * u.deg,
        outer_height=0.6 * u.deg,
        angle=130 * u.deg,
    )
    regions, _ = finder.run(
        region=on_ellipse_annulus,
        center=pointing,
        exclusion_mask=small_mask,
    )
    assert len(regions) == 5


center = SkyCoord(0.5, 0.0, unit="deg")
other_region_finder_param = [
    (RectangleSkyRegion(center, 0.5 * u.deg, 0.5 * u.deg, angle=0 * u.deg), 3),
    (RectangleSkyRegion(center, 0.5 * u.deg, 1 * u.deg, angle=0 * u.deg), 1),
    (RectangleSkyRegion(center, 0.5 * u.deg, 1 * u.deg, angle=90 * u.deg), 1),
    (EllipseSkyRegion(center, 0.1 * u.deg, 1 * u.deg, angle=0 * u.deg), 2),
    (EllipseSkyRegion(center, 0.1 * u.deg, 1 * u.deg, angle=60 * u.deg), 3),
    (EllipseSkyRegion(center, 0.1 * u.deg, 1 * u.deg, angle=90 * u.deg), 2),
]


@pytest.mark.parametrize("region, nreg", other_region_finder_param)
def test_non_circular_regions(region, nreg):
    pointing = SkyCoord(0.0, 0.0, unit="deg")

    finder = ReflectedRegionsFinder(min_distance_input="0 deg")
    regions, _ = finder.run(center=pointing, region=region)
    assert len(regions) == nreg


def test_bad_on_region(exclusion_mask, on_region):
    pointing = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    finder = ReflectedRegionsFinder(
        min_distance_input="0 deg",
    )
    regions, _ = finder.run(
        center=pointing,
        region=on_region,
        exclusion_mask=exclusion_mask,
    )
    assert len(regions) == 0


@requires_data()
def test_reflected_bkg_maker(on_region, reflected_bkg_maker, observations):
    datasets = []

    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_edges(np.logspace(-0.5, 2, 11) * u.TeV, name="energy_true")

    geom = RegionGeom(region=on_region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    maker = SpectrumDatasetMaker(selection=["counts"])

    for obs in observations:
        dataset = maker.run(dataset_empty, obs)
        dataset_on_off = reflected_bkg_maker.run(dataset, obs)
        datasets.append(dataset_on_off)

    assert_allclose(datasets[0].counts_off.data.sum(), 76)
    assert_allclose(datasets[1].counts_off.data.sum(), 60)

    regions_0 = compound_region_to_regions(datasets[0].counts_off.geom.region)
    regions_1 = compound_region_to_regions(datasets[1].counts_off.geom.region)
    assert_allclose(len(regions_0), 11)
    assert_allclose(len(regions_1), 11)


@requires_data()
def test_reflected_bkg_maker_no_off(reflected_bkg_maker, observations, caplog):
    pos = SkyCoord(83.6333313, 21.51444435, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)

    maker = SpectrumDatasetMaker(selection=["counts", "exposure"])

    datasets = []

    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_edges(np.logspace(-0.5, 2, 11) * u.TeV, name="energy_true")
    geom = RegionGeom.create(region=region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)
    for obs in observations:
        dataset = maker.run(dataset_empty, obs)
        dataset_on_off = reflected_bkg_maker.run(dataset, obs)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, obs)
        datasets.append(dataset_on_off)

    assert datasets[0].counts_off is None
    assert_allclose(datasets[0].acceptance_off, 0)
    assert_allclose(datasets[0].mask_safe.data, False)

    assert "WARNING" in [record.levelname for record in caplog.records]

    message1 = (
        f"ReflectedRegionsBackgroundMaker failed. "
        f"No OFF region found outside exclusion mask for dataset '{datasets[0].name}'."
    )
    message2 = (
        f"ReflectedRegionsBackgroundMaker failed. "
        f"Setting {datasets[0].name} mask to False."
    )

    assert message1 in [record.message for record in caplog.records]
    assert message2 in [record.message for record in caplog.records]


@requires_data()
def test_reflected_bkg_maker_no_off_background(reflected_bkg_maker, observations):
    pos = SkyCoord(83.6333313, 21.51444435, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)

    maker = SpectrumDatasetMaker(selection=["counts", "background"])

    datasets = []

    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_edges(np.logspace(-0.5, 2, 11) * u.TeV, name="energy_true")
    geom = RegionGeom.create(region=region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    for obs in observations:
        dataset = maker.run(dataset_empty, obs)
        dataset_on_off = reflected_bkg_maker.run(dataset, obs)
        datasets.append(dataset_on_off)

    assert_allclose(datasets[0].counts_off.data, 0)
    assert_allclose(datasets[0].acceptance_off, 0)


def test_wobble_regions_finder():
    center = SkyCoord(83.6333313, 21.51444435, unit="deg", frame="icrs")
    source_angle = 35 * u.deg

    on_region = CircleSkyRegion(
        center=center,
        radius=0.15 * u.deg,
    )
    pointing = on_region.center.directional_offset_by(
        separation=0.4 * u.deg,
        position_angle=180 * u.deg + source_angle,
    )

    assert u.isclose(
        pointing.position_angle(on_region.center).to(u.deg), source_angle, rtol=0.05
    )
    n_off_regions = 3

    finder = WobbleRegionsFinder(n_off_regions)
    regions, _ = finder.run(on_region, pointing)

    assert len(regions) == 3

    for i, off_region in enumerate(regions, start=1):
        assert u.isclose(pointing.separation(off_region.center), 0.4 * u.deg)
        expected = source_angle + 360 * u.deg * i / (n_off_regions + 1)
        assert u.isclose(
            pointing.position_angle(off_region.center).to(u.deg),
            expected.to(u.deg),
            rtol=0.001,
        )

    # test with exclusion region
    pos = pointing.directional_offset_by(
        separation=0.4 * u.deg,
        position_angle=source_angle + 90 * u.deg,
    )
    exclusion_region = CircleSkyRegion(pos, Angle(0.2, "deg"))
    geom = WcsGeom.create(skydir=pos, binsz=0.01, width=10.0)
    exclusion_mask = ~geom.region_mask([exclusion_region])

    finder = WobbleRegionsFinder(n_off_regions)
    regions, _ = finder.run(on_region, pointing, exclusion_mask)

    assert len(regions) == 2


def test_wobble_regions_finder_overlapping(caplog):
    """Test that overlapping regions are not produced"""
    center = SkyCoord(83.6333313, 21.51444435, unit="deg", frame="icrs")
    source_angle = 35 * u.deg

    on_region = CircleSkyRegion(
        center=center,
        radius=0.5 * u.deg,
    )
    pointing = on_region.center.directional_offset_by(
        separation=0.4 * u.deg,
        position_angle=180 * u.deg + source_angle,
    )

    n_off_regions = 3

    finder = WobbleRegionsFinder(n_off_regions)

    with caplog.at_level(logging.WARNING):
        regions, _ = finder.run(on_region, pointing)

    assert len(regions) == 0
    assert caplog.record_tuples == [
        (
            "gammapy.makers.background.reflected",
            logging.WARNING,
            "Found overlapping off regions, returning no regions",
        )
    ]


@requires_data()
def test_reflected_bkg_maker_with_wobble_finder(
    on_region, observations, exclusion_mask
):
    datasets = []

    reflected_bkg_maker = ReflectedRegionsBackgroundMaker(
        region_finder=WobbleRegionsFinder(n_off_regions=3),
        exclusion_mask=exclusion_mask,
    )

    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")

    geom = RegionGeom(region=on_region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom)

    spectrum_dataset_maker = SpectrumDatasetMaker(selection=["counts"])

    for obs in observations:
        dataset = spectrum_dataset_maker.run(dataset_empty, obs)
        dataset_on_off = reflected_bkg_maker.run(dataset, obs)
        datasets.append(dataset_on_off)

    regions_0 = compound_region_to_regions(datasets[0].counts_off.geom.region)
    regions_1 = compound_region_to_regions(datasets[1].counts_off.geom.region)
    assert_allclose(len(regions_0), 3)
    assert_allclose(len(regions_1), 3)


@requires_data()
def test_reflected_bkg_maker_fixed_rad_max(
    reflected_bkg_maker, observations_fixed_rad_max
):
    e_reco = MapAxis.from_energy_bounds(0.1, 10, 5, unit="TeV")
    e_true = MapAxis.from_energy_bounds(0.1, 10, 5, unit="TeV", name="energy_true")

    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.1414, "deg")
    region = CircleSkyRegion(pos, radius)

    geom = RegionGeom(region=region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    maker = SpectrumDatasetMaker(selection=["counts"])

    obs = observations_fixed_rad_max[0]
    dataset = maker.run(dataset_empty, obs)
    dataset_on_off = reflected_bkg_maker.run(dataset, obs)

    assert_allclose(dataset_on_off.counts_off.data.sum(), 217)

    regions_0 = compound_region_to_regions(dataset_on_off.counts_off.geom.region)
    assert_allclose(len(regions_0), 6)


@requires_data()
def test_reflected_bkg_maker_fixed_rad_max_wobble(
    exclusion_mask, observations_fixed_rad_max
):
    reflected_bkg_maker = ReflectedRegionsBackgroundMaker(
        region_finder=WobbleRegionsFinder(n_off_regions=3),
        exclusion_mask=exclusion_mask,
    )
    e_reco = MapAxis.from_energy_bounds(0.1, 10, 5, unit="TeV")
    e_true = MapAxis.from_energy_bounds(0.1, 10, 5, unit="TeV", name="energy_true")

    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.1414, "deg")
    region = CircleSkyRegion(pos, radius)

    geom = RegionGeom(region=region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    maker = SpectrumDatasetMaker(selection=["counts"])

    obs = observations_fixed_rad_max[0]
    dataset = maker.run(dataset_empty, obs)
    dataset_on_off = reflected_bkg_maker.run(dataset, obs)

    assert_allclose(dataset_on_off.counts_off.data.sum(), 102)

    regions_0 = compound_region_to_regions(dataset_on_off.counts_off.geom.region)
    assert_allclose(len(regions_0), 3)


@requires_data()
def test_reflected_bkg_maker_fixed_rad_max_bad(
    reflected_bkg_maker, observations_fixed_rad_max
):
    e_reco = MapAxis.from_energy_bounds(0.1, 10, 5, unit="TeV")

    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    bad_radius = Angle(0.11, "deg")
    region_bad_size = CircleSkyRegion(pos, bad_radius)

    maker = SpectrumDatasetMaker(selection=["counts"])
    obs = observations_fixed_rad_max[0]

    geom_bad_size = RegionGeom(region=region_bad_size, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom_bad_size)

    dataset = maker.run(dataset_empty, obs)
    with pytest.raises(ValueError):
        reflected_bkg_maker.run(dataset, obs)

    region_bad_shape = RectangleSkyRegion(pos, 0.2 * u.deg, 0.2 * u.deg)
    geom_bad_shape = RegionGeom(region_bad_shape, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom_bad_shape)

    dataset = maker.run(dataset_empty, obs)
    with pytest.raises(ValueError):
        reflected_bkg_maker.run(dataset, obs)

    region_point_shape = PointSkyRegion(pos)
    geom_point_shape = RegionGeom(region_point_shape, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom_point_shape)

    dataset = maker.run(dataset_empty, obs)
    with pytest.raises(TypeError):
        reflected_bkg_maker.run(dataset, obs)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import (
    CircleSkyRegion,
    EllipseAnnulusSkyRegion,
    EllipseSkyRegion,
    RectangleSkyRegion,
)
from gammapy.data import DataStore
from gammapy.maps import WcsGeom, WcsNDMap
from gammapy.spectrum import ReflectedRegionsBackgroundMaker, ReflectedRegionsFinder
from gammapy.spectrum.make import SpectrumDatasetMaker
from gammapy.utils.regions import compound_region_to_list
from gammapy.utils.testing import (
    assert_quantity_allclose,
    mpl_plot_check,
    requires_data,
)


@pytest.fixture(scope="session")
def exclusion_mask():
    """Example mask for testing."""
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    exclusion_region = CircleSkyRegion(pos, Angle(0.3, "deg"))
    geom = WcsGeom.create(skydir=pos, binsz=0.02, width=10.0)
    mask = geom.region_mask([exclusion_region], inside=False)
    return WcsNDMap(geom, data=mask)


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


@pytest.fixture()
def spectrum_dataset_maker(on_region):
    e_reco = np.logspace(0, 2, 5) * u.TeV
    e_true = np.logspace(-0.5, 2, 11) * u.TeV
    return SpectrumDatasetMaker(region=on_region, e_reco=e_reco, e_true=e_true)


@pytest.fixture()
def reflected_bkg_maker(on_region, exclusion_mask):
    return ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)


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
        center=pointing,
        region=on_region,
        exclusion_mask=exclusion_mask,
        min_distance_input="0 deg",
    )
    finder.run()
    regions = finder.reflected_regions
    assert len(regions) == nreg1
    assert_quantity_allclose(regions[3].center.icrs.ra, reg3_ra, rtol=1e-2)

    # Test without exclusion
    finder.exclusion_mask = None
    finder.run()
    regions = finder.reflected_regions
    assert len(regions) == nreg2

    # Test with too small exclusion
    small_mask = exclusion_mask.cutout(pointing, Angle("0.1 deg"))
    finder.exclusion_mask = small_mask
    finder.run()
    regions = finder.reflected_regions
    assert len(regions) == nreg3

    # Test with maximum number of regions
    finder.max_region_number = 5
    finder.run()
    regions = finder.reflected_regions
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
    finder.region = on_ellipse_annulus
    finder.reference_map = None
    finder.run()
    regions = finder.reflected_regions
    assert len(regions) == 5


center = SkyCoord(0.5, 0.0, unit="deg")
other_region_finder_param = [
    (RectangleSkyRegion(center, 0.5 * u.deg, 0.5 * u.deg, angle=0 * u.deg), 3),
    (RectangleSkyRegion(center, 0.5 * u.deg, 1 * u.deg, angle=0 * u.deg), 1),
    (RectangleSkyRegion(center, 0.5 * u.deg, 1 * u.deg, angle=90 * u.deg), 0),
    (EllipseSkyRegion(center, 0.1 * u.deg, 1 * u.deg, angle=0 * u.deg), 2),
    (EllipseSkyRegion(center, 0.1 * u.deg, 1 * u.deg, angle=60 * u.deg), 3),
    (EllipseSkyRegion(center, 0.1 * u.deg, 1 * u.deg, angle=90 * u.deg), 0),
]


@pytest.mark.parametrize("region, nreg", other_region_finder_param)
def test_non_circular_regions(region, nreg):
    pointing = SkyCoord(0.0, 0.0, unit="deg")

    finder = ReflectedRegionsFinder(
        center=pointing, region=region, min_distance_input="0 deg"
    )
    finder.run()
    regions = finder.reflected_regions
    assert len(regions) == nreg


def bad_on_region(exclusion_mask, on_region):
    pointing = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    finder = ReflectedRegionsFinder(
        center=pointing,
        region=on_region,
        exclusion_mask=exclusion_mask,
        min_distance_input="0 deg",
    )
    finder.run()
    regions = finder.reflected_regions
    assert len(regions) == 0

    # try plotting
    with mpl_plot_check():
        finder.plot()


@requires_data()
def test_reflected_bkg_maker(spectrum_dataset_maker, reflected_bkg_maker, observations):
    datasets = []

    for obs in observations:
        dataset = spectrum_dataset_maker.run(obs, selection=["counts"])
        dataset_on_off = reflected_bkg_maker.run(dataset, obs)
        datasets.append(dataset_on_off)

    assert_allclose(datasets[0].counts_off.data.sum(), 76)
    assert_allclose(datasets[1].counts_off.data.sum(), 60)

    regions_0 = compound_region_to_list(datasets[0].counts_off.region)
    regions_1 = compound_region_to_list(datasets[1].counts_off.region)
    assert_allclose(len(regions_0), 11)
    assert_allclose(len(regions_1), 11)

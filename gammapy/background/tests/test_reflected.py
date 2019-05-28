# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ...utils.testing import (
    requires_data,
    requires_dependency,
    assert_quantity_allclose,
    mpl_plot_check,
)
from ...maps import WcsNDMap, WcsGeom
from ...data import DataStore
from ..reflected import ReflectedRegionsFinder, ReflectedRegionsBackgroundEstimator


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


@pytest.fixture(scope="session")
def bkg_estimator(observations, exclusion_mask, on_region):
    """Example background estimator for testing."""
    maker = ReflectedRegionsBackgroundEstimator(
        observations=observations,
        on_region=on_region,
        exclusion_mask=exclusion_mask,
        min_distance_input="0.2 deg",
    )
    maker.run()
    return maker


@requires_data()
def test_find_reflected_regions(exclusion_mask, on_region):
    pointing = SkyCoord(83.2, 22.5, unit="deg")
    fregions = ReflectedRegionsFinder(
        center=pointing,
        region=on_region,
        exclusion_mask=exclusion_mask,
        min_distance_input="0 deg",
    )
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 14
    assert_quantity_allclose(regions[3].center.icrs.ra, Angle("83.674 deg"), rtol=1e-2)

    # Test without exclusion
    fregions.exclusion_mask = None
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 16

    # Test with too small exclusion
    small_mask = exclusion_mask.cutout(pointing, Angle("0.2 deg"))
    fregions.exclusion_mask = small_mask
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 16
    assert_quantity_allclose(regions[3].center.icrs.ra, Angle("83.674 deg"), rtol=1e-2)

    # Test with maximum number of regions
    fregions.max_region_number = 5
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 5


@requires_data()
class TestReflectedRegionBackgroundEstimator:
    def test_basic(self, bkg_estimator):
        assert "ReflectedRegionsBackgroundEstimator" in str(bkg_estimator)

    def test_run(self, bkg_estimator):
        assert len(bkg_estimator.result[1].off_region) == 11
        assert "Reflected" in str(bkg_estimator.result[1])

    @requires_dependency("matplotlib")
    def test_plot(self, bkg_estimator):
        with mpl_plot_check():
            bkg_estimator.plot(idx=1, add_legend=True)
            bkg_estimator.plot(idx=[0, 1])

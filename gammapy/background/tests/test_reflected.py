# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion, EllipseAnnulusSkyRegion
from ...utils.testing import (
    requires_data,
    requires_dependency,
    assert_quantity_allclose,
    mpl_plot_check,
)
from ...maps import WcsNDMap, WcsGeom
from ...data import DataStore
from ..reflected import ReflectedRegionsFinder, ReflectedRegionsBackgroundEstimator

region_finder_param = [ (SkyCoord(83.2, 22.5, unit="deg"), 15, Angle("82.592 deg"), 17, 17),
                        (SkyCoord(84.2, 22.5, unit="deg"), 17, Angle("83.636 deg"), 19, 19),
                        (SkyCoord(83.2, 21.5, unit="deg"), 15, Angle("83.672 deg"), 17, 17),
                        ]

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
@pytest.mark.parametrize(("pointing_pos, nreg1, reg3_ra, nreg2, nreg3"), region_finder_param)
def test_find_reflected_regions(exclusion_mask, on_region, pointing_pos, nreg1, reg3_ra, nreg2, nreg3):
    pointing = pointing_pos
    fregions = ReflectedRegionsFinder(
        center=pointing,
        region=on_region,
        exclusion_mask=exclusion_mask,
        min_distance_input="0 deg",
    )
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == nreg1
    assert_quantity_allclose(regions[3].center.icrs.ra, reg3_ra, rtol=1e-2)

    # Test without exclusion
    fregions.exclusion_mask = None
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == nreg2

    # Test with too small exclusion
    small_mask = exclusion_mask.cutout(pointing, Angle("0.1 deg"))
    fregions.exclusion_mask = small_mask
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == nreg3

    # Test with maximum number of regions
    fregions.max_region_number = 5
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 5

    # Test with an other type of region
    on_ellipse_annulus = EllipseAnnulusSkyRegion(center=on_region.center.transform_to('galactic'),
        inner_width = 0.1 * u.deg, outer_width = 0.2 * u.deg,
        inner_height = 0.3 * u.deg, outer_height = 0.6 * u.deg,
        angle = 130 * u.deg)
    fregions.region = on_ellipse_annulus
    fregions.reference_map = None
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

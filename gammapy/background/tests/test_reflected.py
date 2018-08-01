# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ...utils.testing import requires_data, requires_dependency, assert_quantity_allclose
from ...maps import WcsNDMap
from ...data import DataStore
from ..reflected import ReflectedRegionsFinder, ReflectedRegionsBackgroundEstimator


@pytest.fixture
def mask():
    """Example mask for testing."""
    pos = SkyCoord(83.63, 22.01, unit='deg')
    exclusion_region = CircleSkyRegion(pos, Angle(0.3, 'deg'))
    template_map = WcsNDMap.create(skydir=pos, binsz=0.02, width=10.)
    return template_map.make_region_mask(exclusion_region, inside=False)


@pytest.fixture
def on_region():
    """Example on_region for testing."""
    pos = SkyCoord(83.63, 22.01, unit='deg')
    radius = Angle(0.11, 'deg')
    return CircleSkyRegion(pos, radius)


@pytest.fixture
def obs_list():
    """Example observation list for testing."""
    DATA_DIR = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
    datastore = DataStore.from_dir(DATA_DIR)
    return datastore.obs_list([23523, 23526])


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_find_reflected_regions(mask, on_region):
    pointing = SkyCoord(83.2, 22.5, unit='deg')
    fregions = ReflectedRegionsFinder(
        center=pointing,
        region=on_region,
        exclusion_mask=mask,
        min_distance_input='0 deg',
    )
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 15
    assert_quantity_allclose(regions[3].center.icrs.ra, Angle('83.674 deg'), rtol=1e-2)

    # Test without exclusion
    fregions.exclusion_mask = None
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 16

    # Test with too small exclusion
    small_mask = mask.make_cutout(pointing, Angle('0.2 deg'))[0]
    fregions.exclusion_mask = small_mask
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 16
    assert_quantity_allclose(regions[3].center.icrs.ra, Angle('83.674 deg'), rtol=1e-2)

    # Test with maximum number of regions
    fregions.max_region_number = 5
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 5


@pytest.fixture
def bkg_estimator(obs_list, on_region, mask):
    """Example background estimator for testing."""
    return ReflectedRegionsBackgroundEstimator(
        obs_list=obs_list,
        on_region=on_region,
        exclusion_mask=mask,
    )


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestReflectedRegionBackgroundEstimator:

    @staticmethod
    def test_run(bkg_estimator):
        assert 'ReflectedRegionsBackgroundEstimator' in str(bkg_estimator)
        bkg_estimator.finder.min_distance = Angle('0.2 deg')
        bkg_estimator.run()
        assert len(bkg_estimator.result[1].off_region) == 11
        assert 'Reflected' in str(bkg_estimator.result[1])

    @staticmethod
    @requires_dependency('matplotlib')
    def test_plot(bkg_estimator):
        bkg_estimator.run()
        bkg_estimator.plot()
        bkg_estimator.plot(idx=1)
        bkg_estimator.plot(idx=[0, 1])

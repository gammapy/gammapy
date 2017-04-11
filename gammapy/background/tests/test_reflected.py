# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ..reflected import ReflectedRegionsFinder, ReflectedRegionsBackgroundEstimator
from ...utils.testing import requires_data, requires_dependency
from ...image import SkyImage
from ...data import DataStore
from ..reflected import find_reflected_regions, ReflectedRegionsBackgroundEstimator



@pytest.fixture
def mask():
    """Example mask for testing."""
    filename = '$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits'
    return SkyImage.read(filename, hdu='EXCLUSION')


@pytest.fixture
def on_region():
    """Example on_region for testing."""
    pos = SkyCoord(80.2, 23.5, unit='deg', frame='icrs')
    radius = Angle(0.4, 'deg')
    region = CircleSkyRegion(pos, radius)
    return region


@pytest.fixture
def obs_list():
    """Example observation list for testing."""
    DATA_DIR = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
    datastore = DataStore.from_dir(DATA_DIR)
    obs_ids = [23523, 23526]
    return datastore.obs_list(obs_ids)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_find_reflected_regions(mask, on_region):
    pointing = SkyCoord(83.2, 22.7, unit='deg', frame='icrs')
    finder = ReflectedRegionsFinder(exclusion_mask=mask,
                                    min_distance_input='0 deg')
    finder.run(region=on_region, center=pointing)
    regions = finder.reflected_regions
    assert (len(regions)) == 20
    assert_quantity_allclose(regions[3].center.icrs.ra, Angle('81.752 deg'), rtol=1e-2)

    # Test without exclusion
    finder = ReflectedRegionsFinder()
    finder.run(region=on_region, center=pointing)
    regions = finder.reflected_regions

    # Test with too small exclusion
    pointing = SkyCoord(73.2, 22.7, unit='deg', frame='icrs')
    finder.run(region=on_region, center=pointing)
    regions = finder.reflected_regions
    assert (len(regions)) == 48
    assert_quantity_allclose(regions[3].center.icrs.ra, Angle('78.567 deg'), rtol=1e-2)


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestReflectedRegionBackgroundEstimator:

    def setup(self):
        temp = ReflectedRegionsBackgroundEstimator(on_region=on_region(),
                                                   exclusion=mask(),
        self.obs_list = obs_list()
        self.bg_maker = temp

    def test_basic(self):
        assert 'ReflectedRegionsBackgroundEstimator' in str(self.bg_maker)

    def test_process(self, obs_list, mask):
        bg_estimate = self.bg_maker.process(obs=obs_list[1])
        assert len(bg_estimate.off_region) == 22

    def test_run(self):
        #self.bg_maker.config.update(min_distance = '0.2 deg')
        self.bg_maker.run(self.obs_list)
        result = self.bg_maker.result
        assert len(result[1].off_region) == 22

    @requires_dependency('matplotlib')
    def test_plot(self):
        self.bg_maker.run(self.obs_list)
        result = self.bg_maker.result
        self.bg_maker.plot(result=result, obs_list=self.obs_list)
        self.bg_maker.plot(result=result, obs_list=self.obs_list, idx=[0, 1])

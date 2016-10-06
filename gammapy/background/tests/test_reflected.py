# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ..reflected import find_reflected_regions, ReflectedRegionsBackgroundEstimator
from ...image import SkyMask
from ...utils.testing import requires_data, requires_dependency
from ...data import Target, DataStore

@pytest.fixture
def mask():
    """Example mask for testing."""
    filename = '$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits'
    return SkyMask.read(filename, hdu=1)

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
    regions = find_reflected_regions(region=on_region, center=pointing,
                                     exclusion_mask=mask,
                                     min_distance_input=Angle('0 deg'))
    assert (len(regions)) == 20
    assert_quantity_allclose(regions[3].center.icrs.ra, Angle('81.752 deg'), rtol=1e-2)


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestReflectedRegionBackgroundEstimator:
    def setup(self):
        temp = ReflectedRegionsBackgroundEstimator(on_region = on_region(),
                                                   exclusion = mask(),
                                                   obs_list = obs_list())
        self.bg_maker = temp

    def test_basic(self):
        assert 'ReflectedRegionsBackgroundEstimator' in str(self.bg_maker)

    def test_process(self, on_region, obs_list, mask):
        bg_estimate = self.bg_maker.process(on_region=on_region,
                                            obs=obs_list[1],
                                            exclusion=mask)
        assert len(bg_estimate.off_region) == 22

    def test_run(self):
        self.bg_maker.config.update(min_distance = '0.2 deg')
        self.bg_maker.run()
        assert len(self.bg_maker.result[1].off_region) == 21

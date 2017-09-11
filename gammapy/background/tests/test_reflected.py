# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ...utils.testing import requires_data, requires_dependency
from ...image import SkyImage
from ...data import DataStore
from ..reflected import ReflectedRegionsFinder, ReflectedRegionsBackgroundEstimator


@pytest.fixture
def mask():
    """Example mask for testing."""
    filename = '$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits'
    return SkyImage.read(filename, hdu='EXCLUSION')


@pytest.fixture
def on_region():
    """Example on_region for testing."""
    pos = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle(0.11, 'deg')
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
    pointing = SkyCoord(83.2, 22.5, unit='deg')
    fregions = ReflectedRegionsFinder(center=pointing,
                                      region=on_region,
                                      exclusion_mask=mask,
                                      min_distance_input=Angle('0 deg'))
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
    small_mask = mask.cutout(pointing, Angle('0.2 deg'))
    fregions.exclusion_mask = small_mask
    fregions.run()
    regions = fregions.reflected_regions
    assert len(regions) == 16
    assert_quantity_allclose(regions[3].center.icrs.ra, Angle('83.674 deg'), rtol=1e-2)


@pytest.fixture
def bkg_estimator():
    """Example background estimator for testing."""
    estimator = ReflectedRegionsBackgroundEstimator(obs_list=obs_list(),
                                                    on_region=on_region(),
                                                    exclusion_mask=mask())
    return estimator


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestReflectedRegionBackgroundEstimator:

    def setup(self):
        self.bg_maker = bkg_estimator()

    def test_basic(self):
        assert 'ReflectedRegionsBackgroundEstimator' in str(self.bg_maker)

    def test_run(self):
        self.bg_maker.finder.min_distance = Angle('0.2 deg')
        self.bg_maker.run()
        assert len(self.bg_maker.result[1].off_region) == 11
        assert 'Reflected' in str(self.bg_maker.result[1])

    @requires_dependency('matplotlib')
    def test_plot(self):
        self.bg_maker.run()
        self.bg_maker.plot()
        self.bg_maker.plot(idx=1)
        self.bg_maker.plot(idx=[0, 1])

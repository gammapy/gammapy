# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.tests.helper import pytest, assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ..observation import SpectrumObservation
from ..energy_group import SpectrumEnergyGrouping, calculate_flux_point_binning


@pytest.fixture(scope='session')
def obs():
    """An example SpectrumObservation object for tests."""
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits'
    return SpectrumObservation.read(filename)
    # self.table = self.obs.stats_table()


@pytest.fixture(scope='session')
def seg(obs):
    """An example SpectrumEnergyGrouping object for tests."""
    # Run one example here, to have it available for
    seg = SpectrumEnergyGrouping(obs=obs)
    ebounds = [0.3, 1, 3, 10, 30] * u.TeV,
    seg.compute_groups_fixed(ebounds=ebounds)
    return seg


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestSpectrumEnergyGrouping:
    def _test_fixed(self, seg):
        assert seg.energy_group_id == [1, 2, 3]
        assert_quantity_allclose(seg.energy_bounds, [1, 2, 3])

    def _test_adaptive(self, obs):
        seg = SpectrumEnergyGrouping(obs=obs)
        seg.compute_range_safe()
        seg.compute_groups_adaptive(quantity='sigma', threshold=2.0)

    def test_str(self, seg):
        assert 'Number of groups: 72' in str(seg)

    @requires_dependency('matplotlib')
    def _test_plot(self, seg):
        seg.plot()


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_flux_points_binning():
    obs = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')
    energy_binning = calculate_flux_point_binning(obs_list=[obs], min_signif=3)
    assert_quantity_allclose(energy_binning[5], 1.668 * u.TeV, rtol=1e-3)

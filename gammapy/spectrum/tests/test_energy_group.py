# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.tests.helper import pytest, assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ..observation import SpectrumObservation
from ..energy_group import SpectrumEnergyGroupMaker, calculate_flux_point_binning


@pytest.fixture(scope='session')
def obs():
    """An example SpectrumObservation object for tests."""
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits'
    return SpectrumObservation.read(filename)
    # self.table = self.obs.stats_table()


@pytest.fixture(scope='session')
def seg(obs):
    """An example SpectrumEnergyGroupMaker object for tests."""
    # Run one example here, to have it available for
    seg = SpectrumEnergyGroupMaker(obs=obs)
    ebounds = [0.3, 1, 3, 10, 30] * u.TeV
    seg.compute_range_safe()
    seg.compute_groups_fixed(ebounds=ebounds)
    return seg


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestSpectrumEnergyGrouping:
    def test_str(self, seg):
        assert 'Number of groups: 6' in str(seg)

    def test_fixed(self, seg):
        t = seg.groups.to_group_table()
        assert_equal(t['energy_group_idx'], [0, 1, 2, 3, 4, 5])
        assert_equal(t['bin_idx_min'], [0, 30, 36, 44, 54, 62])
        assert_equal(t['bin_idx_max'], [29, 35, 43, 53, 61, 71])
        assert_equal(t['bin_type'], ['underflow', 'normal', 'normal', 'normal', 'normal', 'overflow'])
        ebounds = [0.01, 0.4641588833612777, 1, 2.7825594022071227, 10, 27.825594022071225, 100]
        assert_allclose(t['energy_min'].value, ebounds[:-1])
        assert_allclose(t['energy_max'].value, ebounds[1:])
        assert_equal(t['energy_group_n_bins'], [30, 6, 8, 10, 8, 10])

    def _test_adaptive(self, obs):
        seg = SpectrumEnergyGroupMaker(obs=obs)
        seg.compute_range_safe()
        seg.compute_groups_adaptive(quantity='sigma', threshold=2.0)

    @requires_dependency('matplotlib')
    def _test_plot(self, seg):
        seg.plot()


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_flux_points_binning():
    obs = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')
    energy_binning = calculate_flux_point_binning(obs_list=[obs], min_signif=3)
    assert_quantity_allclose(energy_binning[5], 1.668 * u.TeV, rtol=1e-3)

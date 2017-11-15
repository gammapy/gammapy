# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
import numpy as np
from astropy.tests.helper import assert_quantity_allclose
import pytest
from ...utils.testing import requires_dependency, requires_data
from ..observation import SpectrumObservation
from ..energy_group import SpectrumEnergyGroupMaker, calculate_flux_point_binning
from ..core import PHACountsSpectrum

@pytest.fixture(scope='session')
def obs():
    """An example SpectrumObservation object for tests."""
    pha_ebounds = np.arange(1, 11) * u.TeV
    on_vector = PHACountsSpectrum(
        energy_lo=pha_ebounds[:-1],
        energy_hi=pha_ebounds[1:],
        data=np.zeros(len(pha_ebounds) - 1),
        meta=dict(EXPOSURE=99)
    )
    return SpectrumObservation(on_vector=on_vector)


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestSpectrumEnergyGrouping:
    def test_str(self, obs):
        seg = SpectrumEnergyGroupMaker(obs=obs)
        ebounds = [1.25, 5.5, 7.5] * u.TeV
        seg.compute_groups_fixed(ebounds=ebounds)
        assert 'Number of groups: 4' in str(seg)

    def test_fixed(self, obs):
        ebounds = [1.25, 5.5, 7.5] * u.TeV
        seg = SpectrumEnergyGroupMaker(obs=obs)
        seg.compute_groups_fixed(ebounds=ebounds)
        t = seg.groups.to_group_table()
        assert_equal(t['energy_group_idx'], [0, 1, 2, 3])
        assert_equal(t['bin_idx_min'], [0, 1, 4, 6])
        assert_equal(t['bin_idx_max'], [0, 3, 5, 8])
        assert_equal(t['bin_type'], ['underflow', 'normal', 'normal', 'overflow'])

        ebounds = [1.0, 2.0, 5.0, 7.0, 10.0] * u.TeV
        assert_allclose(t['energy_min'], ebounds[:-1])
        assert_allclose(t['energy_max'], ebounds[1:])
        assert_equal(t['energy_group_n_bins'], [1, 3, 2, 3])

    def test_edges(self, obs):
        ebounds = [2, 5, 7] * u.TeV
        seg = SpectrumEnergyGroupMaker(obs=obs)
        seg.compute_groups_fixed(ebounds=ebounds)

        # We want thoses conditions verified
        t = seg.groups.to_group_table()
        assert_equal(len(t), 4)
        assert_equal(t['bin_type'], ['underflow', 'normal', 'normal', 'overflow'])
        assert_equal(t['bin_idx_min'], [0, 1, 4, 6])
        assert_equal(t['bin_idx_max'], [0, 3, 5, 8])
        assert_allclose(t['energy_min'], [1, 2, 5, 7] * u.TeV)
        assert_allclose(t['energy_max'], [2, 5, 7, 10] * u.TeV)
        assert_equal(t['energy_group_n_bins'], [1, 3, 2, 3])
        
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
    assert_quantity_allclose(energy_binning[5], 2.448 * u.TeV, rtol=1e-3)

    

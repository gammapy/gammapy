# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
import numpy as np
from astropy.tests.helper import assert_quantity_allclose
import pytest
from ...utils.testing import requires_dependency, requires_data
from ..observation import SpectrumObservation
from ..energy_group import (
    SpectrumEnergyGroups, SpectrumEnergyGroupMaker,
    SpectrumEnergyGroup, calculate_flux_point_binning)
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
class TestSpectrumEnergyGroupMaker:

    def test_fixed(self, obs):
        seg = SpectrumEnergyGroupMaker(obs=obs)

        ebounds = [1.25, 5.5, 7.5] * u.TeV
        seg.compute_groups_fixed(ebounds=ebounds)
        t = seg.groups.to_group_table()

        assert_equal(t['energy_group_idx'], [0, 1, 2, 3])
        assert_equal(t['bin_idx_min'], [0, 1, 4, 6])
        assert_equal(t['bin_idx_max'], [0, 3, 5, 8])
        assert_equal(t['bin_type'], ['underflow', 'normal', 'normal', 'overflow'])

        ebounds = [1.0, 2.0, 5.0, 7.0, 10.0] * u.TeV
        assert_allclose(t['energy_min'], ebounds[:-1])
        assert_allclose(t['energy_max'], ebounds[1:])

    def test_edges(self, obs):
        seg = SpectrumEnergyGroupMaker(obs=obs)

        ebounds = [2, 5, 7] * u.TeV
        seg.compute_groups_fixed(ebounds=ebounds)

        # We want thoses conditions verified
        t = seg.groups.to_group_table()
        assert_equal(len(t), 4)
        assert_equal(t['bin_type'], ['underflow', 'normal', 'normal', 'overflow'])
        assert_equal(t['bin_idx_min'], [0, 1, 4, 6])
        assert_equal(t['bin_idx_max'], [0, 3, 5, 8])
        assert_allclose(t['energy_min'], [1, 2, 5, 7] * u.TeV)
        assert_allclose(t['energy_max'], [2, 5, 7, 10] * u.TeV)

    def _test_adaptive(self, obs):
        seg = SpectrumEnergyGroupMaker(obs=obs)
        seg.compute_range_safe()
        seg.compute_groups_adaptive(quantity='sigma', threshold=2.0)


class TestSpectrumEnergyGroup:

    def setup(self):
        self.group = SpectrumEnergyGroup(3, 10, 20, 'normal', 100 * u.TeV, 200 * u.TeV)

    def test_init(self):
        """Establish argument order in `__init__` and attributes."""
        g = self.group
        assert g.energy_group_idx == 3
        assert g.bin_idx_min == 10
        assert g.bin_idx_max == 20
        assert g.bin_type == 'normal'
        assert g.energy_min == 100 * u.TeV
        assert g.energy_max == 200 * u.TeV

    def test_repr(self):
        txt = ("SpectrumEnergyGroup(energy_group_idx=3, bin_idx_min=10, bin_idx_max=20, "
               "bin_type='normal', energy_min=<Quantity 100.0 TeV>, energy_max=<Quantity 200.0 TeV>)")
        assert repr(self.group) == txt

    def test_to_dict(self):
        # Check that it round-trips
        d = self.group.to_dict()
        g = SpectrumEnergyGroup(**d)
        assert g == self.group

    def test_bin_idx_array(self):
        assert_equal(self.group.bin_idx_array, np.arange(10, 21))

    def test_bin_table(self):
        t = self.group.bin_table
        assert_equal(t['bin_idx'], np.arange(10, 21))
        assert_equal(t['energy_group_idx'], 3)
        assert_equal(t['bin_type'], 'normal')

    def test_contains_energy(self):
        energy = [99, 100, 199, 200] * u.TeV
        actual = self.group.contains_energy(energy)
        expected = [False, True, True, False]
        assert_equal(actual, expected)


# TODO: remove this: instead only use the example from `TestSpectrumEnergyGroups.setup` for all tests.
@pytest.fixture()
def groups(obs):
    table = obs.stats_table()
    table['bin_idx'] = np.arange(len(table))
    table['energy_group_idx'] = table['bin_idx']
    return SpectrumEnergyGroups.from_total_table(table)


class TestSpectrumEnergyGroups:

    def setup(self):
        G = SpectrumEnergyGroup
        # An example without under- and overflow bins
        self.groups = SpectrumEnergyGroups([
            # energy_group_idx, bin_idx_min, bin_idx_max, bin_type, energy_min, energy_max
            G(0, 10, 20, 'normal', 100 * u.TeV, 210 * u.TeV),
            G(1, 21, 25, 'normal', 210 * u.TeV, 260 * u.TeV),
            G(5, 26, 26, 'normal', 260 * u.TeV, 270 * u.TeV),
            G(6, 27, 30, 'normal', 270 * u.TeV, 300 * u.TeV),
        ])

    def test_repr(self):
        assert repr(self.groups) == 'SpectrumEnergyGroups(len=4)'

    def test_str(self):
        txt = str(self.groups)
        assert 'SpectrumEnergyGroups' in txt
        assert 'energy_group_idx' in txt

    def test_group_table(self):
        """Check that info to and from group table round-trips"""
        t = self.groups.to_group_table()
        g = SpectrumEnergyGroups.from_group_table(t)
        assert g == self.groups

    @pytest.mark.xfail
    def test_total_table(self):
        """Check that info to and from total table round-trips"""
        t = self.groups.to_total_table()
        # The energy_min and energy_max aren't available to fill in `to_total_table`,
        # because they aren'

        g = SpectrumEnergyGroups.from_total_table(t)
        assert g == self.groups

    def test_find_list_idx(self):
        assert self.groups.find_list_idx(energy=270 * u.TeV) == 3  # On the edge
        assert self.groups.find_list_idx(energy=271 * u.TeV) == 3  # inside a bin

        with pytest.raises(IndexError):
            self.groups.find_list_idx(energy=99 * u.TeV)  # too low

        with pytest.raises(IndexError):
            self.groups.find_list_idx(energy=300 * u.TeV)  # too high, left edge is not inclusive

    def test_make_and_replace_merged_group(self, groups):
        # Merge first 4 bins
        groups.make_and_replace_merged_group(0, 3, 'underflow')
        assert_equal(groups[0].bin_type, 'underflow')
        assert_equal(groups[0].bin_idx_min, 0)
        assert_equal(groups[0].bin_idx_max, 3)
        assert_equal(groups[0].energy_group_idx, 0)

        # Flag 5th bin as normal
        groups.make_and_replace_merged_group(1, 1, 'normal')
        assert_equal(groups[1].bin_type, 'normal')
        assert_equal(groups[1].energy_group_idx, 1)

        # Merge last 4 bins
        groups.make_and_replace_merged_group(2, 5, 'overflow')
        assert_equal(groups[2].bin_type, 'overflow')
        assert_equal(groups[2].bin_idx_min, 5)
        assert_equal(groups[2].bin_idx_max, 8)
        assert_equal(groups[2].energy_group_idx, 2)
        assert_equal(len(groups), 3)

    def test_flag_and_merge_out_of_range(self, groups):
        ebounds = [2, 5, 7] * u.TeV
        groups.flag_and_merge_out_of_range(ebounds)

        t = groups.to_total_table()
        assert_equal(t['bin_type'], ['underflow', 'normal', 'normal', 'normal',
                                     'normal', 'normal', 'overflow', 'overflow', 'overflow'])
        assert_equal(t['energy_group_idx'], [0, 1, 2, 3, 4, 5, 6, 6, 6])

    def test_apply_energy_binning(self, groups):
        ebounds = [2, 5, 7] * u.TeV
        groups.apply_energy_binning(ebounds)

        t = groups.to_total_table()
        assert_equal(t['energy_group_idx'], [0, 1, 1, 1, 2, 2, 3, 4, 5])
        assert_equal(t['bin_idx'], [0, 1, 2, 3, 4, 5, 6, 7, 8])
        assert_equal(t['bin_type'], ['normal', 'normal', 'normal', 'normal',
                                     'normal', 'normal', 'normal', 'normal', 'normal'])


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_flux_points_binning():
    obs = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')
    energy_binning = calculate_flux_point_binning(obs_list=[obs], min_signif=3)
    assert_quantity_allclose(energy_binning[5], 2.448 * u.TeV, rtol=1e-3)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_data, requires_dependency
from ..core import PHACountsSpectrum
from ..observation import SpectrumObservation, SpectrumObservationList
from ..energy_group import SpectrumEnergyGroups, SpectrumEnergyGroupMaker, SpectrumEnergyGroup


class TestSpectrumEnergyGroup:

    @pytest.fixture()
    def group(self):
        return SpectrumEnergyGroup(3, 10, 20, 'normal', 100 * u.TeV, 200 * u.TeV)

    def test_init(self, group):
        """Establish argument order in `__init__` and attributes."""
        assert group.energy_group_idx == 3
        assert group.bin_idx_min == 10
        assert group.bin_idx_max == 20
        assert group.bin_type == 'normal'
        assert group.energy_min == 100 * u.TeV
        assert group.energy_max == 200 * u.TeV

    def test_repr(self, group):
        assert 'SpectrumEnergyGroup' in repr(group)

    def test_to_dict(self, group):
        # Check that it round-trips
        group2 = SpectrumEnergyGroup(**group.to_dict())
        assert group2 == group

    def test_bin_idx_array(self, group):
        assert_equal(group.bin_idx_array, np.arange(10, 21))

    def test_bin_table(self, group):
        table = group.bin_table
        assert_equal(table['bin_idx'], np.arange(10, 21))
        assert_equal(table['energy_group_idx'], 3)
        assert_equal(table['bin_type'], 'normal')

    def test_contains_energy(self, group):
        energy = [99, 100, 199, 200] * u.TeV
        actual = group.contains_energy(energy)
        expected = [False, True, True, False]
        assert_equal(actual, expected)


class TestSpectrumEnergyGroups:

    @pytest.fixture()
    def groups(self):
        return SpectrumEnergyGroups([
            # energy_group_idx, bin_idx_min, bin_idx_max, bin_type, energy_min, energy_max
            SpectrumEnergyGroup(0, 10, 20, 'normal', 100 * u.TeV, 210 * u.TeV),
            SpectrumEnergyGroup(1, 21, 25, 'normal', 210 * u.TeV, 260 * u.TeV),
            SpectrumEnergyGroup(5, 26, 26, 'normal', 260 * u.TeV, 270 * u.TeV),
            SpectrumEnergyGroup(6, 27, 30, 'normal', 270 * u.TeV, 300 * u.TeV),
        ])

    def test_repr(self, groups):
        assert repr(groups) == 'SpectrumEnergyGroups(len=4)'

    def test_str(self, groups):
        txt = str(groups)
        assert 'SpectrumEnergyGroups' in txt
        assert 'energy_group_idx' in txt

    def test_copy(self, groups):
        """Make sure groups.copy() is a deep copy"""
        groups2 = groups.copy()
        groups2[0].bin_type == 'spam'
        assert groups[0].bin_type == 'normal'

    def test_group_table(self, groups):
        """Check that info to and from group table round-trips"""
        table = groups.to_group_table()
        groups2 = SpectrumEnergyGroups.from_group_table(table)
        assert groups2 == groups

    @pytest.mark.xfail
    def test_total_table(self, groups):
        """Check that info to and from total table round-trips"""
        table = groups.to_total_table()
        # The energy_min and energy_max aren't available to fill in `to_total_table`,
        # because they aren'
        groups2 = SpectrumEnergyGroups.from_total_table(table)
        assert groups2 == groups

    def test_energy_range(self, groups):
        actual = groups.energy_range
        expected = [100, 300] * u.TeV
        assert_allclose(actual, expected)
        assert actual.unit == 'TeV'

    def test_energy_bounds(self, groups):
        actual = groups.energy_bounds
        expected = [100, 210, 260, 270, 300] * u.TeV
        assert_allclose(actual, expected)
        assert actual.unit == 'TeV'

    def test_find_list_idx(self, groups):
        assert groups.find_list_idx(energy=270 * u.TeV) == 3  # On the edge
        assert groups.find_list_idx(energy=271 * u.TeV) == 3  # inside a bin

        with pytest.raises(IndexError):
            groups.find_list_idx(energy=99 * u.TeV)  # too low

        with pytest.raises(IndexError):
            groups.find_list_idx(energy=300 * u.TeV)  # too high, left edge is not inclusive

    def test_make_and_replace_merged_group(self, groups):
        groups.make_and_replace_merged_group(0, 1, 'underflow')
        groups.make_and_replace_merged_group(1, 1, 'normal')
        groups.make_and_replace_merged_group(2, 2, 'overflow')

        expected = SpectrumEnergyGroups([
            SpectrumEnergyGroup(0, 10, 25, 'underflow', 100 * u.TeV, 260 * u.TeV),
            SpectrumEnergyGroup(1, 26, 26, 'normal', 260 * u.TeV, 270 * u.TeV),
            SpectrumEnergyGroup(2, 27, 30, 'overflow', 270 * u.TeV, 300 * u.TeV),
        ])

        assert groups == expected

    def test_apply_energy_min_and_max(self, groups):
        groups.apply_energy_min(210 * u.TeV)
        groups.apply_energy_max(260 * u.TeV)

        expected = SpectrumEnergyGroups([
            SpectrumEnergyGroup(0, 10, 20, 'underflow', 100 * u.TeV, 210 * u.TeV),
            SpectrumEnergyGroup(1, 21, 25, 'normal', 210 * u.TeV, 260 * u.TeV),
            SpectrumEnergyGroup(2, 26, 30, 'overflow', 260 * u.TeV, 300 * u.TeV),
        ])

        assert groups == expected

    @pytest.mark.xfail(reason='apply_energy_binning is still buggy at the upper ebounds edge '
                       'as well as for ebounds bins that are very small, i.e. have no bin!')
    def test_apply_energy_binning_same_ebounds(self, groups):
        """Grouping with original ebounds should leave groups unchanged."""
        ebounds = groups.energy_bounds
        ebounds[-1] = 299 * u.TeV
        groups2 = groups.copy()
        groups2.apply_energy_binning(ebounds)
        print(ebounds)
        print(groups)
        print(groups2)
        assert groups == groups2

    def test_apply_energy_binning_different_ebounds(self):
        # TODO: Use the same groups fixture for this test as for all other tests.
        # (if changes are needed there to cover all interesting behaviour, do it there!)
        groups = SpectrumEnergyGroups([
            SpectrumEnergyGroup(i, i, i, 'normal', (i + 1) * u.TeV, (i + 2) * u.TeV)
            for i in range(9)
        ])

        ebounds = [2, 5, 7] * u.TeV
        groups.apply_energy_binning(ebounds)

        expected = SpectrumEnergyGroups([
            SpectrumEnergyGroup(0, 0, 0, 'normal', 1 * u.TeV, 2 * u.TeV),
            SpectrumEnergyGroup(1, 1, 3, 'normal', 2 * u.TeV, 5 * u.TeV),
            SpectrumEnergyGroup(2, 4, 5, 'normal', 5 * u.TeV, 7 * u.TeV),
            SpectrumEnergyGroup(3, 6, 6, 'normal', 7 * u.TeV, 8 * u.TeV),
            SpectrumEnergyGroup(4, 7, 7, 'normal', 8 * u.TeV, 9 * u.TeV),
            SpectrumEnergyGroup(5, 8, 8, 'normal', 9 * u.TeV, 10 * u.TeV),
        ])

        assert groups == expected


class TestSpectrumEnergyGroupMaker:

    @pytest.fixture(scope='session')
    def obs(self):
        """An example SpectrumObservation object for tests."""
        pha_ebounds = np.arange(1, 11) * u.TeV
        on_vector = PHACountsSpectrum(
            energy_lo=pha_ebounds[:-1],
            energy_hi=pha_ebounds[1:],
            data=np.zeros(len(pha_ebounds) - 1),
            meta=dict(EXPOSURE=99)
        )
        return SpectrumObservation(on_vector=on_vector)

    @pytest.mark.parametrize('ebounds', [
        [1.25, 5.5, 7.5] * u.TeV,
        [2, 5, 7] * u.TeV,
    ])
    def test_groups_fixed(self, obs, ebounds):
        seg = SpectrumEnergyGroupMaker(obs=obs)
        seg.compute_groups_fixed(ebounds=ebounds)
        groups = seg.groups

        expected = SpectrumEnergyGroups([
            SpectrumEnergyGroup(0, 0, 0, 'underflow', 1 * u.TeV, 2 * u.TeV),
            SpectrumEnergyGroup(1, 1, 3, 'normal', 2 * u.TeV, 5 * u.TeV),
            SpectrumEnergyGroup(2, 4, 5, 'normal', 5 * u.TeV, 7 * u.TeV),
            SpectrumEnergyGroup(3, 6, 8, 'overflow', 7 * u.TeV, 10 * u.TeV),
        ])

        assert groups == expected

    @requires_dependency('scipy')
    @requires_data('gammapy-extra')
    def test_adaptive(self):
        obs = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')
        obs = SpectrumObservationList([obs]).stack()

        seg = SpectrumEnergyGroupMaker(obs=obs)
        seg.compute_range_safe()
        energy_binning = seg.compute_groups_adaptive(min_signif=3)

        # energy_binning = calculate_flux_point_binning(obs_list=[obs], min_signif=3)
        assert_quantity_allclose(energy_binning[5], 2.448 * u.TeV, rtol=1e-3)

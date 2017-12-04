# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectrum energy bin grouping.

There are three classes:

* SpectrumEnergyGroup - one group
* SpectrumEnergyGroups - one grouping, i.e. collection of groups
* SpectrumEnergyGroupMaker - algorithms to compute groupings.

Algorithms to compute groupings are both on SpectrumEnergyGroups and SpectrumEnergyGroupMaker.
The difference is that SpectrumEnergyGroups contains the algorithms and book-keeping that
just have to do with the groups, whereas SpectrumEnergyGroupMaker also accesses
information from SpectrumObservation (e.g. safe energy range or counts data) and
implements higher-level algorithms.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from ..extern.six.moves import UserList
from astropy.units import Quantity
from astropy.table import Table
from astropy.table import vstack as table_vstack
from ..utils.table import table_from_row_data, table_row_to_dict
from ..data import ObservationStats

__all__ = [
    'SpectrumEnergyGroup',
    'SpectrumEnergyGroups',
    'SpectrumEnergyGroupMaker',
]

# TODO: improve the code so that this isn't needed!
INVALID_GROUP_INDEX = -99

# TODO: this is used for input at the moment,
# but for output the `bin_type` field is used.
# Make up your mind!
UNDERFLOW_BIN_INDEX = -1
OVERFLOW_BIN_INDEX = -2


class SpectrumEnergyGroup(object):
    """Spectrum energy group.

    Represents a consecutive range of bin indices (both ends inclusive).
    """
    fields = [
        'energy_group_idx', 'bin_idx_min', 'bin_idx_max',
        'bin_type', 'energy_min', 'energy_max',
    ]
    """List of data members of this class."""

    valid_bin_types = ['normal', 'underflow', 'overflow']
    """Valid values for ``bin_types`` attribute."""

    def __init__(self, energy_group_idx, bin_idx_min, bin_idx_max, bin_type,
                 energy_min, energy_max):
        self.energy_group_idx = energy_group_idx
        self.bin_idx_min = bin_idx_min
        self.bin_idx_max = bin_idx_max
        if bin_type not in self.valid_bin_types:
            raise ValueError('Invalid bin type: {}'.format(bin_type))
        self.bin_type = bin_type
        self.energy_min = energy_min
        self.energy_max = energy_max

    @classmethod
    def from_dict(cls, data):
        data = dict((_, data[_]) for _ in cls.fields)
        return cls(**data)

    @property
    def _data(self):
        return [(_, getattr(self, _)) for _ in self.fields]

    def __repr__(self):
        txt = ['{}={!r}'.format(k, v) for k, v in self._data]
        return '{}({})'.format(self.__class__.__name__, ', '.join(txt))

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        return OrderedDict(self._data)

    @property
    def bin_idx_array(self):
        """Numpy array of bin indices in the group."""
        return np.arange(self.bin_idx_min, self.bin_idx_max + 1)

    @property
    def bin_table(self):
        """Create `~astropy.table.Table` with bins in the group.

        Columns are: ``energy_group_idx``, ``bin_idx``, ``bin_type``
        """
        table = Table()
        table['bin_idx'] = self.bin_idx_array
        table['energy_group_idx'] = self.energy_group_idx
        table['bin_type'] = self.bin_type
        return table

    def contains_energy(self, energy):
        """Does this group contain a given energy?"""
        return (self.energy_min <= energy) & (energy < self.energy_max)


class SpectrumEnergyGroups(UserList):
    """List of `~gammapy.spectrum.SpectrumEnergyGroup` objects.

    A helper class used by the `gammapy.spectrum.SpectrumEnergyMaker`.
    """

    def __repr__(self):
        return '{}(len={})'.format(self.__class__.__name__, len(self))

    def __str__(self):
        ss = '{}:\n'.format(self.__class__.__name__)
        lines = self.to_group_table().pformat(max_width=-1, max_lines=-1)
        ss += '\n'.join(lines)
        return ss + '\n'

    def copy(self):
        """Deep copy"""
        return deepcopy(self)

    @classmethod
    def from_total_table(cls, table):
        """Create list of SpectrumEnergyGroup objects from table."""
        groups = cls()

        for energy_group_idx in np.unique(table['energy_group_idx']):
            mask = table['energy_group_idx'] == energy_group_idx
            group_table = table[mask]
            bin_idx_min = group_table['bin_idx'][0]
            bin_idx_max = group_table['bin_idx'][-1]
            # bin_type = group_table['bin_type']
            if energy_group_idx == UNDERFLOW_BIN_INDEX:
                bin_type = 'underflow'
            elif energy_group_idx == OVERFLOW_BIN_INDEX:
                bin_type = 'overflow'
            else:
                bin_type = 'normal'
            energy_min = group_table['energy_min'].quantity[0]
            energy_max = group_table['energy_max'].quantity[-1]

            group = SpectrumEnergyGroup(
                energy_group_idx=energy_group_idx,
                bin_idx_min=bin_idx_min,
                bin_idx_max=bin_idx_max,
                bin_type=bin_type,
                energy_min=energy_min,
                energy_max=energy_max,
            )
            groups.append(group)

        return groups

    @classmethod
    def from_group_table(cls, table):
        """Create from energy groups in `~astropy.table.Table` format."""
        return cls([
            SpectrumEnergyGroup.from_dict(table_row_to_dict(row))
            for row in table
        ])

    def to_total_table(self):
        """Table with one energy bin per row (`~astropy.table.Table`).

        Columns:

        * ``energy_group_idx`` - Energy group index (int)
        * ``bin_idx`` - Energy bin index (int)
        * ``bin_type`` - Bin type {'normal', 'underflow', 'overflow'} (str)

        There are no energy columns, because the per-bin energy info
        was lost during grouping.
        """
        tables = [group.bin_table for group in self]
        return table_vstack(tables)

    def to_group_table(self):
        """Table with one energy group per row (`~astropy.table.Table`).

        Columns:

        * ``energy_group_idx`` - Energy group index (int)
        * ``energy_group_n_bins`` - Number of bins in the energy group (int)
        * ``bin_idx_min`` - First bin index in the energy group (int)
        * ``bin_idx_max`` - Last bin index in the energy group (int)
        * ``bin_type`` - Bin type {'normal', 'underflow', 'overflow'} (str)
        * ``energy_min`` - Energy group start energy (Quantity)
        * ``energy_max`` - Energy group end energy (Quantity)
        """
        rows = [group.to_dict() for group in self]
        table = table_from_row_data(rows)
        return table

    @property
    def bin_idx_range(self):
        """Tuple (left, right) with range of bin indices (both edges inclusive)."""
        left = self[0].bin_idx_min
        right = self[-1].bin_idx_max
        return left, right

    @property
    def energy_range(self):
        """Total energy range (`~astropy.units.Quantity` of length 2)."""
        return Quantity([self[0].energy_min, self[-1].energy_max])

    @property
    def energy_bounds(self):
        """Energy group bounds (`~astropy.units.Quantity`)."""
        energy = [_.energy_min for _ in self]
        energy.append(self[-1].energy_max)
        return Quantity(energy)

    def find_list_idx(self, energy):
        """Find the list index corresponding to a given energy."""
        for idx, group in enumerate(self):
            if group.contains_energy(energy):
                return idx

            # TODO: do we need / want this behaviour?
            # If yes, could add via a kwarg `last_bin_right_edge_inclusive=False`
            # For last energy group
            # if idx == len(self) - 1 and energy == group.energy_max:
            #     return idx

        raise IndexError('No group found with energy: {}'.format(energy))

    def find_list_idx_range(self, energy_min, energy_max):
        """TODO: document.

        * Min index is the bin that contains ``energy_range.min``
        * Max index is the bin that is below the one that contains ``energy_range.max``
        * This way we don't loose any bins or count them twice.
        * Containment is checked for each bin as [min, max)
        """
        idx_min = self.find_list_idx(energy=energy_min)
        idx_max = self.find_list_idx(energy=energy_max) - 1
        return idx_min, idx_max

    def make_and_replace_merged_group(self, list_idx_min, list_idx_max, bin_type):
        """Merge energy groups and update indexes"""
        # Create a merged group object
        group = self.make_merged_group(
            list_idx_min=list_idx_min,
            list_idx_max=list_idx_max,
            bin_type=bin_type,
        )

        # Delete previous groups
        [self.pop(list_idx_min) for _ in range(list_idx_max - list_idx_min + 1)]

        # Insert the merged group
        self.insert(list_idx_min, group)
        self.reindex_groups()
        return self

    def reindex_groups(self):
        """Re-index groups"""
        for energy_group_idx, group in enumerate(self):
            group.energy_group_idx = energy_group_idx

    def make_merged_group(self, list_idx_min, list_idx_max, bin_type):
        """Merge group according to indexes"""
        left_group = self[list_idx_min]
        right_group = self[list_idx_max]

        return SpectrumEnergyGroup(
            energy_group_idx=INVALID_GROUP_INDEX,
            bin_idx_min=left_group.bin_idx_min,
            bin_idx_max=right_group.bin_idx_max,
            bin_type=bin_type,
            energy_min=left_group.energy_min,
            energy_max=right_group.energy_max,
        )

    # TODO: choose one of the apply energy min / max methods!

    def apply_energy_min_old(self, energy):
        """Modify list in-place to apply a min energy cut."""
        idx_max = self.find_list_idx(energy)
        self.make_and_replace_merged_group(0, idx_max, 'underflow')

    def apply_energy_min(self, energy):
        t = self.to_group_table()
        idx_max = np.where(t['energy_min'].quantity < energy)[0][-1]
        self.make_and_replace_merged_group(0, idx_max, 'underflow')

    def apply_energy_max_old(self, energy):
        """Modify list in-place to apply a max energy cut."""
        idx_min = self.find_list_idx(energy)
        idx_max = len(self) - 1
        self.make_and_replace_merged_group(idx_min, idx_max, 'overflow')

    def apply_energy_max(self, energy):
        t = self.to_group_table()
        idx_min = np.where(t['energy_max'].quantity > energy)[0][0]
        self.make_and_replace_merged_group(idx_min, len(self) - 1, 'overflow')

    def clip_to_valid_range(self, list_idx):
        """TODO: document"""
        if self[list_idx].bin_type == 'underflow':
            list_idx += 1

        if self[list_idx].bin_type == 'overflow':
            list_idx -= 1

        if list_idx < 0:
            raise IndexError('list_idx {} < 0'.format(list_idx))
        if list_idx >= len(self):
            raise IndexError('list_idx {} > len(self)'.format(list_idx))

        return list_idx

    def apply_energy_binning(self, ebounds):
        """Apply an energy binning."""

        for idx in range(len(ebounds) - 1):
            energy_min = ebounds[idx]
            energy_max = ebounds[idx + 1]
            list_idx_min, list_idx_max = self.find_list_idx_range(energy_min, energy_max)

            # Be sure to leave underflow and overflow bins alone
            # TODO: this is pretty ugly ... make it better somehow!
            list_idx_min = self.clip_to_valid_range(list_idx_min)
            list_idx_max = self.clip_to_valid_range(list_idx_max)

            self.make_and_replace_merged_group(
                list_idx_min=list_idx_min,
                list_idx_max=list_idx_max,
                bin_type='normal',
            )


class SpectrumEnergyGroupMaker(object):
    """Energy bin groups for spectral analysis.

    This class contains both methods that run algorithms
    that compute groupings as well as the results as data members
    and methods to debug and assess the results.

    The input ``obs`` is used read-only, to access the counts energy
    binning, as well as some other info that is used for energy bin grouping.

    This class creates the ``groups`` attribute on construction,
    with exactly one group per energy bin. It is then modified by calling
    methods on this class, usually to declare some bins as under- and
    overflow (i.e. not to be used in spectral analysis), and to group
    bins (e.g. for flux point computation).

    See :ref:`spectrum_energy_group` for examples.

    Parameters
    ----------
    obs : `~gammapy.spectrum.SpectrumObservation`
        Spectrum observation

    Attributes
    ----------
    obs : `~gammapy.spectrum.SpectrumObservation`
        Spectrum observation data
    groups : `~gammapy.spectrum.SpectrumEnergyGroups`
        List of energy groups

    See also
    --------
    SpectrumEnergyGroups, SpectrumEnergyGroup, FluxPointEstimator
    """

    def __init__(self, obs):
        self.obs = obs
        self.groups = self._groups_from_obs(obs)

    @staticmethod
    def _groups_from_obs(obs):
        """Compute energy groups list with one group per energy bin.

        Parameters
        ----------
        obs : `~gammapy.spectrum.SpectrumObservation`
            Spectrum observation data

        Returns
        -------
        groups : `~gammapy.spectrum.SpectrumEnergyGroups`
            List of energy groups
        """
        # Start with a table with the obs energy binning
        table = obs.stats_table()
        # Make one group per bin
        table['bin_idx'] = np.arange(len(table))
        table['energy_group_idx'] = np.arange(len(table))
        return SpectrumEnergyGroups.from_total_table(table)

    def compute_range_safe(self):
        """Apply safe energy range of observation to ``groups``.

        This method takes the safe energy range information from ``self.obs``
        and changes ``self.groups`` like this:

        * group bins below the safe energy range into one group of type "underflow"
        * group bins above the safe energy range into one group of type "overflow"
        """
        bins = self.obs.on_vector.bins_in_safe_range

        underflow = bins[0] - 1

        # If no low threshold is set no underflow bin is needed
        if underflow >= 0:
            self.groups.make_and_replace_merged_group(0, underflow, 'underflow')

        # The group binning has changed
        overflow = bins[-1] - underflow
        max_bin = self.groups[-1].energy_group_idx

        # If no high threshold is set no overflow bin is needed
        if overflow <= max_bin:
            self.groups.make_and_replace_merged_group(overflow, max_bin, 'overflow')

    def compute_groups_fixed(self, ebounds):
        """Apply grouping for a given fixed energy binning.

        Parameters
        ----------
        ebounds : `~astropy.units.Quantity`
            Energy bounds array
        """
        self.groups.apply_energy_min(energy=ebounds[0])
        self.groups.apply_energy_max(energy=ebounds[-1])
        self.groups.apply_energy_binning(ebounds=ebounds)

    def compute_groups_adaptive(self, min_signif, rebin_factor=2):
        """Compute energy binning for flux points.

        This is useful to get an energy binning to use with
        :func:`~gammapy.spectrum.FluxPoints` Each bin in the
        resulting energy binning will include a ``min_signif`` source detection.

        TODO: It is required that at least two fine bins be included in one
        flux point interval, otherwise the sherpa covariance method breaks
        down.

        Parameters
        ----------
        min_signif : float
            Required significance for each bin
        """
        obs = self.obs
        # NOTE: Results may vary from FitSpectrum since there the rebin
        # parameter can only have fixed values, here it grows linearly. Also it
        # has to start at 2 here (see docstring)

        # First first bin above low threshold and last bin below high threshold
        current_ebins = obs.on_vector.energy
        current_bin = (current_ebins.find_node(obs.lo_threshold) + 1)[0]
        max_bin = (current_ebins.find_node(obs.hi_threshold))[0]

        # List holding final energy binning
        binning = [current_ebins.lo[current_bin]]

        # Precompute ObservationStats for each bin
        obs_stats = [obs.stats(i) for i in range(current_ebins.nbins)]
        while current_bin + rebin_factor <= max_bin:
            # Merge bins together until min_signif is reached
            stats_list = obs_stats[current_bin:current_bin + rebin_factor:1]
            stats = ObservationStats.stack(stats_list)
            sigma = stats.sigma
            if sigma < min_signif or np.isnan(sigma):
                rebin_factor += 1
                continue

            # Append upper bin edge of good energy bin to final binning
            binning.append(current_ebins.lo[current_bin + rebin_factor])
            current_bin += rebin_factor

        binning = Quantity(binning)
        # Replace highest bin edge by high threshold
        binning[-1] = obs.hi_threshold

        # TODO: fill self.groups instead of returning the binning!
        # self.groups.apply_energy_binning(binning)

        return binning

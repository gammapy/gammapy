# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The main class is ``SpectrumEnergyGroupMaker``

These are helper classes to implement the grouping algorithms,
they are not part of the public Gammapy API at the moment:

* ``EnergyRange``
* ``SpectrumEnergyGroup``

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from ..extern.six.moves import UserList
from astropy.units import Quantity
from astropy.table import Table
from ..utils.fits import table_from_row_data
from ..data import ObservationStats
from .observation import SpectrumObservationList
import logging

__all__ = [
    'SpectrumEnergyGroupMaker',
    'SpectrumEnergyGroup',
    'SpectrumEnergyGroups',
]

# TODO: improve the code so that this isn't needed!
INVALID_GROUP_INDEX = -99

# TODO: this is used for input at the moment,
# but for output the `bin_type` field is used.
# Make up your mind!
UNDERFLOW_BIN_INDEX = -1
OVERFLOW_BIN_INDEX = -2

log = logging.getLogger(__name__)

class SpectrumEnergyGroupMaker(object):
    """Energy bin groups for spectral analysis.

    This class contains both methods that run algorithms
    that compute groupings as well as the results as data members
    and methods to debug and assess the results.

    The input ``obs`` should be used read-only.
    All computations modify ``table`` or other properties.

    TODO: should this class have a `SpectrumEnergyGroups` data member
    instead of or in addition to `table`?

    See :ref:`spectrum_energy_group` for examples.

    Parameters
    ----------
    obs : `~gammapy.spectrum.SpectrumObservation`
        Spectrum observation

    Attributes
    ----------
    obs : `~gammapy.spectrum.SpectrumObservation`
        Spectrum observation data
    table : `~astropy.table.QTable`
        Table with some per-energy bin stats info.
    groups : `~gammapy.spectrum.SpectrumEnergyGroups`
        List of energy groups.
    """

    def __init__(self, obs):
        self.obs = obs

        # Start with a table with the energy binning and basic stats
        self.table = self.obs.stats_table()
        # Start out with one bin = one group
        self.table['bin_idx'] = np.arange(len(self.table))
        self.table['energy_group_idx'] = self.table['bin_idx']

        # The algorithms use this `groups` object
        # Unfortunately one has to be careful to copy the results
        # back into the `table` object to keep the two in sync
        self.groups = SpectrumEnergyGroups.from_total_table(self.table)

    def __str__(self):
        ss = self.__class__.__name__

        ss += '\nSpectrum table:\n'
        ss += 'Number of bins: {}\n'.format(len(self.table))
        ss += 'Bin range: {}\n'.format((0, len(self.table) - 1))
        ss += 'Energy range: {}\n'.format(self.table_energy_range)

        ss += str(self.groups)

        return ss

    # Properties concerning the total, un-grouped spectrum
    @property
    def table_energy_range(self):
        """Total spectrum energy range (no grouping or range applied)"""
        emin = self.table['energy_min'][0]
        emax = self.table['energy_max'][-1]
        return EnergyRange(emin, emax)

    # Properties for the grouped spectrum
    @property
    def n_groups(self):
        """Number of groups."""
        return len(self.groups)

    # Methods to compute total ranges
    def compute_range_safe(self):
        """Apply safe energy range of observation to ``groups``."""
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

    def set_energy_range(self, emin=None, emax=None):
        """Apply energy range to ``groups``."""
        if emin:
            self.groups.apply_energy_min(emin)
        if emax:
            self.groups.apply_energy_max(emax)

    # Methods to compute groupings
    def compute_groups_fixed(self, ebounds):
        """Compute grouping for a given fixed energy binning."""
        self.groups.flag_and_merge_out_of_range(ebounds=ebounds)
        self.groups.apply_energy_binning(ebounds=ebounds)


class SpectrumEnergyGroup(object):
    """Spectrum energy group.

    Represents a consecutive range of bin indices (both ends inclusive).
    """
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

    def to_dict(self):
        """Convert to `~collections.OrderedDict`."""
        data = OrderedDict()
        data['energy_group_idx'] = self.energy_group_idx
        data['bin_idx_min'] = self.bin_idx_min
        data['bin_idx_max'] = self.bin_idx_max
        data['bin_type'] = self.bin_type
        data['energy_min'] = self.energy_min
        data['energy_max'] = self.energy_max
        return data

    @property
    def bin_idx_range(self):
        """Range of bin indices (both sides inclusive)."""
        return self.bin_idx_min, self.bin_idx_max

    @property
    def energy_range(self):
        """Energy range."""
        return EnergyRange(min=self.energy_min, max=self.energy_max)

    @property
    def bin_idx_list(self):
        """List of bin indices in the group."""
        left, right = self.bin_idx_range
        return list(range(left, right + 1))

    def __str__(self):
        return str(self.to_dict())


class SpectrumEnergyGroups(UserList):
    """List of `~gammapy.spectrum.SpectrumEnergyGroup` objects.

    A helper class used by the `gammapy.spectrum.SpectrumEnergyMaker`.
    """

    def __str__(self):
        ss = 'SpectrumEnergyGroups:\n'
        ss += '\nInfo including underflow- and overflow bins:\n'
        ss += 'Number of groups: {}\n'.format(len(self))
        ss += 'Bin range: {}\n'.format(self.bin_idx_range)
        ss += 'Energy range: {}\n'.format(self.energy_range)
        return ss

    @classmethod
    def from_total_table(cls, table):
        """Create list of SpectrumEnergyGroup objects from table."""
        groups = cls()

        for energy_group_idx in np.unique(table['energy_group_idx']):
            mask = table['energy_group_idx'] == energy_group_idx
            group_table = table[mask]
            bin_idx_min = group_table['bin_idx'][0]
            bin_idx_max = group_table['bin_idx'][-1]
            if energy_group_idx == UNDERFLOW_BIN_INDEX:
                bin_type = 'underflow'
            elif energy_group_idx == OVERFLOW_BIN_INDEX:
                bin_type = 'overflow'
            else:
                bin_type = 'normal'
            energy_min = group_table['energy_min'][0]
            energy_max = group_table['energy_max'][-1]

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
    def from_groups_table(cls, table):
        """Create from energy groups in `~astropy.table.Table` format."""
        groups = cls()
        for row in table:
            group = SpectrumEnergyGroup(
                energy_group_idx=row['energy_group_idx'],
                bin_idx_min=row['bin_idx_min'],
                bin_idx_max=row['bin_idx_max'],
                bin_type=row['bin_type'],
                energy_min=row['energy_min'],
                energy_max=row['energy_max'],
            )
            groups.append(group)

        return groups

    def to_total_table(self):
        """Table with one energy bin per row (`~astropy.table.QTable`).

        Columns:

        * ``energy_group_idx`` - Energy group index (int)
        * ``bin_idx`` - Energy bin index (int)
        * ``bin_type`` - Bin type {'normal', 'underflow', 'overflow'} (str)

        There are no energy columns, because the per-bin energy info
        was lost during grouping.
        """
        rows = []
        for group in self:
            for bin_idx in group.bin_idx_list:
                row = OrderedDict()
                row['energy_group_idx'] = group.energy_group_idx
                row['bin_idx'] = bin_idx
                row['bin_type'] = group.bin_type
                rows.append(row)

        names = ['energy_group_idx', 'bin_idx', 'bin_type']
        return Table(rows=rows, names=names)

    def to_group_table(self):
        """Table with one energy group per row (`~astropy.table.QTable`).

        Columns:

        * ``energy_group_idx`` - Energy group index (int)
        * ``energy_group_n_bins`` - Number of bins in the energy group (int)
        * ``bin_idx_min`` - First bin index in the energy group (int)
        * ``bin_idx_max`` - Last bin index in the energy group (int)
        * ``bin_type`` - Bin type {'normal', 'underflow', 'overflow'} (str)
        * ``energy_min`` - Energy group start energy (Quantity)
        * ``energy_max`` - Energy group end energy (Quantity)
        * ``energy_group_n_bins`` - Number of energy bins in the energy group (int)
        * ``log10_energy_width`` - Energy group width: ``log10(energy_max / energy_min)`` (float)
        """
        rows = [group.to_dict() for group in self]
        table = table_from_row_data(rows)
        table['energy_group_n_bins'] = table['bin_idx_max'] - table['bin_idx_min'] + 1
        table['log10_energy_width'] = np.log10(table['energy_max'] / table['energy_min'])
        return table

    @property
    def bin_idx_range(self):
        """Range of bin indices (left and right inclusive)."""
        left = self[0].bin_idx_min
        right = self[-1].bin_idx_max
        return left, right

    @property
    def energy_range(self):
        """Energy range."""
        return EnergyRange(min=self[0].energy_min, max=self[-1].energy_max)

    def find_list_idx(self, energy):
        """Find the list index corresponding to a given energy."""
        for idx, group in enumerate(self):
            # For last energy group
            if idx == len(self) - 1 and energy == group.energy_max:
                return idx

            if energy in group.energy_range:
                return idx

        raise IndexError('No group found with energy: {}'.format(energy))

    def find_list_idx_range(self, energy_range):
        """TODO: document.

        * Min index is the bin that contains ``energy_range.min``
        * Max index is the bin that is below the one that contains ``energy_range.max``
        * This way we don't loose any bins or count them twice.
        * Containment is checked for each bin as [min, max)
        """
        idx_min = self.find_list_idx(energy=energy_range.min)
        idx_max = self.find_list_idx(energy=energy_range.max) - 1
        return idx_min, idx_max

    def apply_energy_min(self, energy):
        """Modify list in-place to apply a min energy cut."""
        idx_min = 0
        idx_max = self.find_list_idx(energy)
        self.make_and_replace_merged_group(idx_min, idx_max, bin_type='underflow')
        
    def apply_energy_max(self, energy):
        """Modify list in-place to apply a max energy cut."""
        idx_min = self.find_list_idx(energy)
        idx_max = len(self) - 1
        self.make_and_replace_merged_group(idx_min, idx_max, bin_type='overflow')
        
    def apply_energy_binning(self, ebounds):
        """Apply an energy binning.
        
        Before application of the energy binning, overflow
        and underflow bins are flaged. After application of
        energy binning, u/o bins are merged
        """
        for energy_range in EnergyRange.list_from_ebounds(ebounds):
            list_idx_min, list_idx_max = self.find_list_idx_range(energy_range)

            # Be sure to leave underflow and overflow bins alone
            # TODO: this is pretty ugly ... make it better somehow!
            list_idx_min = self.clip_to_valid_range(list_idx_min)
            list_idx_max = self.clip_to_valid_range(list_idx_max)

            self.make_and_replace_merged_group(
                list_idx_min=list_idx_min,
                list_idx_max=list_idx_max,
                bin_type='normal',
            )

    def clip_to_valid_range(self, list_idx):
        """TODO: document"""
        if self[list_idx].bin_type == 'underflow':
            list_idx += 1

        if self[list_idx].bin_type == 'overflow':
            list_idx -= 1

        if list_idx < 0:
            raise IndexError('list_idx {} < 0'.format(list_idx))
        if list_idx >= len(self):
            raise IndexError('list_idx {} > len(self) {}'.format(list_idx))

        return list_idx

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

    def flag_and_merge_out_of_range(self, ebounds):
        """Flag underflow and overflow bins, merge them afterwards"""
        t = self.to_group_table()
        idx_u = np.where(t['energy_min'] < ebounds[0])[0]
        idx_o = np.where(t['energy_max'] > ebounds[-1])[0]
        for idx in idx_u:
            self[idx].bin_type = 'underflow'
        self.make_and_replace_merged_group(
            list_idx_min=idx_u[0],
            list_idx_max=idx_u[-1],
            bin_type='underflow',
        )
        
        for idx in idx_o:
            self[idx].bin_type = 'overflow'
        self.make_and_replace_merged_group(
            list_idx_min=idx_o[0],
            list_idx_max=idx_o[-1],
            bin_type='overflow',
        )
        


                
class EnergyRange(object):
    """Energy range.

    This is just a little helper class.
    We could have used length-2 tuple or Quantity for this.

    TODO: Merge with `~gammapy.utils.energy.EnergyBounds`
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    @property
    def width(self):
        """Energy range width."""
        return self.max - self.min

    @property
    def log10_width(self):
        """Log10 width (sometimes called "dex").
        """
        return np.log10(self.max / self.min)

    @property
    def log_center(self):
        """Log center."""
        return np.sqrt(self.min * self.max)

    def __contains__(self, energy):
        if (self.min <= energy) and (energy < self.max):
            return True
        else:
            return False

    def __repr__(self):
        fmt = 'EnergyRange(min={min}, max={max})'
        return fmt.format(min=self.min, max=self.max)

    @classmethod
    def list_from_ebounds(cls, ebounds):
        """Create list of ``EnergyRange`` from array of energy bounds.

        Examples
        --------
        >>> import astropy.units as u
        >>> from gammapy.spectrum.energy_group import EnergyRange
        >>> ebounds = [0.3, 1, 3, 10] * u.TeV
        >>> EnergyRange.list_from_ebounds(ebounds)
        [EnergyRange(min=0.3 TeV, max=1.0 TeV),
         EnergyRange(min=1.0 TeV, max=3.0 TeV),
         EnergyRange(min=3.0 TeV, max=10.0 TeV)]
        """
        return [
            EnergyRange(min=emin, max=emax)
            for (emin, emax) in zip(ebounds[:-1], ebounds[1:])
        ]


def calculate_flux_point_binning(obs_list, min_signif):
    """Compute energy binning for flux points.

    This is useful to get an energy binning to use with
    :func:`~gammapy.spectrum.FluxPoints` Each bin in the
    resulting energy binning will include a ``min_signif`` source detection.

    TODO: It is required that at least two fine bins be included in one
    flux point interval, otherwise the sherpa covariance method breaks
    down.

    TODO: Refactor, add back to docs

    Parameters
    ----------
    obs_list : `~gammapy.spectrum.SpectrumObservationList`
        Observations
    min_signif : float
        Required significance for each bin

    Returns
    -------
    binning : `~astropy.units.Quantity`
        Energy binning
    """
    # NOTE: Results may vary from FitSpectrum since there the rebin
    # parameter can only have fixed values, here it grows linearly. Also it
    # has to start at 2 here (see docstring)

    # rebin_factor = 1
    rebin_factor = 2

    obs = SpectrumObservationList(obs_list).stack()

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

    return binning

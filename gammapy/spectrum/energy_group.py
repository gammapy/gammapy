# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The main class is ``SpectrumEnergyGrouping``

These are helper classes to implement the grouping algorithms,
they are not part of the public Gammapy API at the moment:

* ``EnergyRange``
* ``SpectrumEnergyGroup``

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.units import Quantity
from astropy.table import Table
from ..utils.fits import table_from_row_data
from ..utils.energy import EnergyBounds
from ..data import ObservationStats
from .observation import SpectrumObservation

__all__ = [
    'SpectrumEnergyGrouping',
    'calculate_flux_point_binning',
]

UNDERFLOW_BIN_INDEX = -1
OVERFLOW_BIN_INDEX = -2


class SpectrumEnergyGrouping(object):
    """Energy bin groups for spectral analysis.

    This class contains both methods that run algorithms
    that compute groupings as well as the results as data members
    and methods to debug and assess the results.

    The input ``obs`` should be used read-only.
    All computations modify ``table`` or other properties.

    TODO: should this class have a `SpectrumEnergyGroupList` data member
    instead of or in addition to `table`?

    See :ref:`spectrum_energy_group` for examples.
    """

    def __init__(self, obs=None):
        self.obs = obs

        # Start with a table with the energy binning and basic stats
        self.table = self.obs.stats_table()
        # Start out with one bin = one group
        self.table['bin_idx'] = np.arange(len(self.table))
        self.table['energy_group_idx'] = self.table['bin_idx']

        # The algorithms use this `groups` object
        # Unfortunately one has to be careful to copy the results
        # back into the `table` object to keep the two in sync
        self.groups = SpectrumEnergyGroupList.from_total_table(self.table)

    def __str__(self):
        """Print a little report"""
        ss = 'SpectrumEnergyGrouping:\n'

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
        """Apply safe energy range to ``groups``.
        """
        emin = self.obs.lo_threshold
        emax = self.obs.hi_threshold
        self.set_energy_range(emin=emin, emax=emax)

    def set_energy_range(self, emin=None, emax=None):
        """Apply energy range to ``groups``.
        """
        if emin:
            self.groups.apply_energy_min(emin)
        if emax:
            self.groups.apply_energy_max(emax)

    # Methods to compute groupings

    def compute_groups_fixed(self, ebounds):
        """Compute grouping for a given fixed energy binning.
        """
        # self.groups.apply_energy_min(energy=ebounds[0])
        # self.groups.apply_energy_max(energy=ebounds[-1])
        # self.groups.apply_energy_binning(ebounds=ebounds)

    def compute_groups_npoints(self, npoints):
        emin, emax = self.get_safe_range()
        npoints = self.config['n_points']
        return EnergyBounds.equal_log_spacing(
            emin=emin, emax=emax, nbins=npoints,
        )


class SpectrumEnergyGroup(object):
    """Spectrum energy group.

    Represents a consecutive range of bin indices (both ends inclusive).
    """
    valid_bin_types = ['normal', 'underflow', 'overflow']

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
        return str(self.as_dict())


class SpectrumEnergyGroupList(list):
    """List of ``SpectrumEnergyGroup`` objects.
    """

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
        """TODO: document"""
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
        """TODO: document"""
        # TODO: add energy_min and energy_max here?
        rows = []
        for group in self:
            for bin_idx in group.bin_idx_list:
                row = OrderedDict()
                row['energy_group_idx'] = group.energy_group_index
                row['bin_idx'] = bin_idx
                row['bin_type'] = group.bin_type
                rows.append(row)
        names = ['energy_group_idx', 'bin_idx', 'bin_type']
        return Table(rows=rows, names=names)

    def to_group_table(self):
        """Convert list of EnergyGroups objects to table."""
        rows = [group.to_dict() for group in self]
        # names = ['energy_group_idx', 'bin_idx_min', 'bin_idx_max', 'bin_type', 'energy_min', 'energy_max']
        return table_from_row_data(rows)

    def get_ids(self, with_underflow=False, with_overflow=False):
        """TODO: document"""
        ids = []
        for group in self:
            if not with_underflow and group.type == 'underflow':
                continue
            if not with_overflow and group.type == 'overflow':
                continue
            ids.append(group.energy_group_idx)

        return ids

    @property
    def bin_idx_range(self):
        """Range of bin indices (left and right inclusive)"""
        left = self[0].bin_idx_min
        right = self[-1].bin_idx_max
        return left, right

    @property
    def energy_range(self):
        """Energy range."""
        return EnergyRange(min=self[0].energy_min, max=self[-1].energy_max)

    def __str__(self):
        ss = 'SpectrumEnergyGroupList:\n'
        ss += 'Number of groups: {}\n'.format(len(self))
        ss += 'Bin range: {}\n'.format(self.bin_idx_range)
        ss += 'Energy range: {}\n'.format(self.energy_range)
        return ss


class EnergyRange(object):
    """Energy range.
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    # @property
    # def width(self):
    #     """Width."""
    #     return self.max - self.min
    #
    # @property
    # def log_center(self):
    #     """Log center."""
    #     return np.sqrt(self.min * self.max)

    def __repr__(self):
        fmt = 'EnergyRange(min={min}, max={max})'
        return fmt.format(min=self.min, max=self.max)


def calculate_flux_point_binning(obs_list, min_signif):
    """Compute energy binning.

    This is useful to get an energy binning to use with
    :func:`~gammapy.spectrum.DifferentialFluxPoints.compute` Each bin in the
    resulting energy binning will include a ``min_signif`` source detection.

    TODO: It is required that at least two fine bins be included in one
    flux point interval, otherwise the sherpa covariance method breaks
    down.

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

    obs = SpectrumObservation.stack(obs_list)

    # First first bin above low threshold and last bin below high threshold
    current_ebins = obs.on_vector.energy
    current_bin = (current_ebins.find_node(obs.lo_threshold) + 1)[0]
    max_bin = (current_ebins.find_node(obs.hi_threshold))[0]

    # List holding final energy binning
    binning = [current_ebins.data[current_bin]]

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
        binning.append(current_ebins.data[current_bin + rebin_factor])
        current_bin += rebin_factor

    binning = Quantity(binning)
    # Replace highest bin edge by high threshold
    binning[-1] = obs.hi_threshold

    return binning

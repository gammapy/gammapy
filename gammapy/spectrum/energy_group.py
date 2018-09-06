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
import logging
from ..extern.six.moves import UserList  # pylint:disable=import-error
from astropy.units import Quantity
from astropy.table import Table
from astropy.table import vstack as table_vstack
from ..utils.table import table_from_row_data, table_row_to_dict

__all__ = ["SpectrumEnergyGroup", "SpectrumEnergyGroups", "SpectrumEnergyGroupMaker"]

log = logging.getLogger(__name__)


class SpectrumEnergyGroup(object):
    """Spectrum energy group.

    Represents a consecutive range of bin indices (both ends inclusive).
    """

    fields = [
        "energy_group_idx",
        "bin_idx_min",
        "bin_idx_max",
        "bin_type",
        "energy_min",
        "energy_max",
    ]
    """List of data members of this class."""

    valid_bin_types = ["normal", "underflow", "overflow"]
    """Valid values for ``bin_types`` attribute."""

    def __init__(
        self,
        energy_group_idx,
        bin_idx_min,
        bin_idx_max,
        bin_type,
        energy_min,
        energy_max,
    ):
        self.energy_group_idx = energy_group_idx
        self.bin_idx_min = bin_idx_min
        self.bin_idx_max = bin_idx_max
        if bin_type not in self.valid_bin_types:
            raise ValueError("Invalid bin type: {}".format(bin_type))
        self.bin_type = bin_type
        self.energy_min = Quantity(energy_min)
        self.energy_max = Quantity(energy_max)

    @classmethod
    def from_dict(cls, data):
        data = dict((_, data[_]) for _ in cls.fields)
        return cls(**data)

    @property
    def _data(self):
        return [(_, getattr(self, _)) for _ in self.fields]

    def __repr__(self):
        txt = ["{}={!r}".format(k, v) for k, v in self._data]
        return "{}({})".format(self.__class__.__name__, ", ".join(txt))

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
        table["bin_idx"] = self.bin_idx_array
        table["energy_group_idx"] = self.energy_group_idx
        table["bin_type"] = self.bin_type
        table["energy_min"] = self.energy_min
        table["energy_max"] = self.energy_max
        return table


class SpectrumEnergyGroups(UserList):
    """List of `~gammapy.spectrum.SpectrumEnergyGroup` objects.

    A helper class used by the `gammapy.spectrum.SpectrumEnergyGroupsMaker`.
    """

    def __repr__(self):
        return "{}(len={})".format(self.__class__.__name__, len(self))

    def __str__(self):
        ss = "{}:\n".format(self.__class__.__name__)
        lines = self.to_group_table().pformat(max_width=-1, max_lines=-1)
        ss += "\n".join(lines)
        return ss + "\n"

    def copy(self):
        """Deep copy"""
        return deepcopy(self)

    @classmethod
    def from_total_table(cls, table):
        """Create list of SpectrumEnergyGroup objects from table."""
        groups = cls()

        for energy_group_idx in np.unique(table["energy_group_idx"]):
            mask = table["energy_group_idx"] == energy_group_idx
            group_table = table[mask]
            bin_idx_min = group_table["bin_idx"][0]
            bin_idx_max = group_table["bin_idx"][-1]
            if len(set(group_table["bin_type"])) > 1:
                raise ValueError("Inconsistent bin_type within group.")
            bin_type = group_table["bin_type"][0]
            energy_min = group_table["energy_min"].quantity[0]
            energy_max = group_table["energy_max"].quantity[-1]

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
        return cls(
            [SpectrumEnergyGroup.from_dict(table_row_to_dict(row)) for row in table]
        )

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
    def energy_range(self):
        """Total energy range (`~astropy.units.Quantity` of length 2)."""
        return Quantity([self[0].energy_min, self[-1].energy_max])

    @property
    def energy_bounds(self):
        """Energy group bounds (`~astropy.units.Quantity`)."""
        energy = [_.energy_min for _ in self]
        energy.append(self[-1].energy_max)
        return Quantity(energy)


class SpectrumEnergyGroupMaker(object):
    """Energy bin groups for spectral analysis.

    This class contains both methods that run algorithms
    that compute groupings as well as the results as data members
    and methods to debug and assess the results.

    The input ``obs`` is used read-only, to access the counts energy
    binning, as well as some other info that is used for energy bin grouping.

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
        self.groups = None

    def groups_from_obs(self):
        """Compute energy groups with one group per energy bin."""
        ebounds_obs = self.obs.e_reco
        size = ebounds_obs.nbins
        table = Table()
        table["bin_idx"] = np.arange(size)
        table["energy_group_idx"] = np.arange(size)
        table["bin_type"] = ["normal"] * size
        table["energy_min"] = ebounds_obs.lower_bounds
        table["energy_max"] = ebounds_obs.upper_bounds
        self.groups = SpectrumEnergyGroups.from_total_table(table)

    def compute_groups_fixed(self, ebounds):
        """Apply grouping for a given fixed energy binning.

        This groups the observation ``obs.e_reco`` binning and
        ``ebounds`` using a nearest neighbor match on the bin edges.

        Parameters
        ----------
        ebounds : `~astropy.units.Quantity`
            Energy bounds array
        """
        ebounds_src = self.obs.e_reco.to(ebounds.unit)
        bin_edges_src = np.arange(len(ebounds_src))

        temp = np.interp(ebounds, ebounds_src, bin_edges_src)
        bin_edges = np.round(temp, decimals=0).astype(np.int)

        # Check for duplicates
        duplicates_removed = set(bin_edges)
        if len(duplicates_removed) != len(bin_edges):
            warn_str = "Input binning\n{}\n contains bins that are finer than the"
            warn_str += " target binning\n{}\n or outside the valid range"
            log.warning(warn_str.format(ebounds, ebounds_src))
        bin_edges = sorted(duplicates_removed)

        # Create normal bins
        groups = []
        for idx in np.arange(len(bin_edges) - 1):
            group = SpectrumEnergyGroup(
                energy_group_idx=-1,
                bin_idx_min=bin_edges[idx],
                bin_idx_max=bin_edges[idx + 1] - 1,
                bin_type="normal",
                energy_min=ebounds_src[bin_edges[idx]],
                energy_max=ebounds_src[bin_edges[idx + 1]],
            )
            groups.append(group)

        if groups == []:
            err_str = "Input binning\n{}\n has no overlap with"
            err_str += " target binning\n{}"
            raise ValueError(err_str.format(ebounds, ebounds_src))

        # Add underflow bin
        start_edge = groups[0].bin_idx_min
        if start_edge != 0:
            underflow = SpectrumEnergyGroup(
                energy_group_idx=-1,
                bin_idx_min=0,
                bin_idx_max=start_edge - 1,
                bin_type="underflow",
                energy_min=ebounds_src[0],
                energy_max=ebounds_src[start_edge],
            )
            groups.insert(0, underflow)

        # Add overflow bin
        end_edge = groups[-1].bin_idx_max
        if end_edge != ebounds_src.nbins - 1:
            overflow = SpectrumEnergyGroup(
                energy_group_idx=-1,
                bin_idx_min=end_edge + 1,
                bin_idx_max=ebounds_src.nbins - 1,
                bin_type="overflow",
                energy_min=ebounds_src[end_edge + 1],
                energy_max=ebounds_src[-1],
            )
            groups.append(overflow)

        # Set energy_group_idx
        for group_idx, group in enumerate(groups):
            group.energy_group_idx = group_idx

        self.groups = SpectrumEnergyGroups(groups)

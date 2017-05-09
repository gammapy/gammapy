# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table, Column, vstack
from astropy.io import ascii
from astropy.units import Quantity
from ..extern.pathlib import Path

__all__ = [
    'ObservationGroupAxis',
    'ObservationGroups',
]


class ObservationGroups(object):
    """Observation groups.

    Class to define observation groups useful for organizing observation
    lists into groups of observations with similar properties.

    The properties and their binning are specified via
    `~gammapy.data.ObservationGroupAxis` objects.

    The class takes as input a list of `~gammapy.data.ObservationGroupAxis`
    objects and defines one group for each possible combination of the
    bins defined in all axes (cartesian product).
    The groups are identified by a unique ``GROUP_ID`` int value.

    The definitions of the groups are internally stored as a
    `~astropy.table.Table` object, the ``obs_groups_table`` member.

    The axis parameters should be either dimensionless or castable
    into `~astropy.units.Quantity` objects.

    For details on the grouping of observations in a list, please
    refer to the `~gammapy.data.ObservationGroups.apply` method.

    See also :ref:`obs_observation_grouping`.

    Parameters
    ----------
    axes : list of `~gammapy.data.ObservationGroupAxis`
        List of observation group axes.

    Examples
    --------
    Define an observation grouping:

    .. code:: python

        alt = Angle([0, 30, 60, 90], 'deg')
        az = Angle([-90, 90, 270], 'deg')
        ntels = np.array([3, 4])
        obs_groups = ObservationGroups([
            ObservationGroupAxis('ALT', alt, fmt='edges'),
            ObservationGroupAxis('AZ', az, fmt='edges'),
            ObservationGroupAxis('N_TELS', ntels, fmt='values'),
        ])

    Print the observation group table (group definitions):

    >>> print(obs_groups.obs_groups_table)

    Print the observation group axes:

    >>> print(obs_groups.info)

    Group the observations of an observation list and print them:

    >>> obs_table_grouped = obs_groups.apply(obs_table)
    >>> print(obs_table_grouped)

    Get the observations of a particular group and print them:

    >>> obs_table_group8 = obs_groups.get_group_of_observations(obs_table_grouped, 8)
    >>> print(obs_table_group8)
    """

    def __init__(self, axes):
        self.axes = axes
        self.obs_groups_table = ObservationGroups.axes_to_table(axes)

    @property
    def n_groups(self):
        """Number of groups (int)."""
        return len(self.obs_groups_table)

    @property
    def list_of_groups(self):
        """List of groups (`~numpy.ndarray`)."""
        return self.obs_groups_table['GROUP_ID'].data

    @staticmethod
    def axes_to_table(axes):
        """Fill the observation group axes into a table.

        Define one row for each possible combination of the
        observation group axis bins. Each row will represent
        an observation group.

        Parameters
        ----------
        axes : `~gammapy.data.ObservationGroupAxis`
            List of observation group axes.

        Returns
        -------
        table : `~astropy.table.Table`
            Table containing the observation group definitions.
        """
        # define table column data
        column_data_min = []
        column_data_max = []
        # loop over observation axes
        for i_axis in range(len(axes)):
            if axes[i_axis].fmt == 'values':
                column_data_min.append(axes[i_axis].bins)
                column_data_max.append(axes[i_axis].bins)
            elif axes[i_axis].fmt == 'edges':
                column_data_min.append(axes[i_axis].bins[:-1])
                column_data_max.append(axes[i_axis].bins[1:])

        # define grids of column data
        ndim = len(axes)
        s0 = (1,) * ndim
        expanding_arrays = [x.reshape(s0[:i] + (-1,) + s0[i + 1::])
                            for i, x in enumerate(column_data_min)]
        column_data_expanded_min = np.broadcast_arrays(*expanding_arrays)
        expanding_arrays = [x.reshape(s0[:i] + (-1,) + s0[i + 1::])
                            for i, x in enumerate(column_data_max)]
        column_data_expanded_max = np.broadcast_arrays(*expanding_arrays)

        # recover units
        for i_dim in range(ndim):
            column_data_expanded_min[i_dim] = _recover_units(column_data_expanded_min[i_dim],
                                                             column_data_min[i_dim])
            column_data_expanded_max[i_dim] = _recover_units(column_data_expanded_max[i_dim],
                                                             column_data_max[i_dim])

        # Make the table
        table = Table()
        for i_axis in range(len(axes)):
            if axes[i_axis].fmt == 'values':
                table[axes[i_axis].name] = column_data_expanded_min[i_axis].flatten()
            elif axes[i_axis].fmt == 'edges':
                table[axes[i_axis].name + "_MIN"] = column_data_expanded_min[i_axis].flatten()
                table[axes[i_axis].name + "_MAX"] = column_data_expanded_max[i_axis].flatten()

        ObservationGroups._add_group_id(table, axes)

        return table

    @staticmethod
    def _add_group_id(table, axes):
        # Compute number of groups
        n_groups = 1
        for i_axis in range(len(axes)):
            n_groups *= axes[i_axis].n_bins

        # fill table, with first the obs group IDs, then the axis columns
        group_id = Column(name='GROUP_ID', data=np.arange(n_groups))
        table.add_column(group_id, index=0)

        return table

    @staticmethod
    def table_to_axes(table):
        """Define observation group axis list from a table.

        Interpret the combinations of bins from a table of groups
        in order to define the corresponding observation group axes.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table containing the observation group definitions.

        Returns
        -------
        axes : `~gammapy.data.ObservationGroupAxis`
            List of observation group axes.
        """
        # subset table: remove obs groups column
        if table.colnames[0] == 'GROUP_ID':
            table = table[table.colnames[1:]]

        axes = []
        for i_col, col_name in enumerate(table.columns):
            data = np.unique(table[col_name].data)
            # recover units
            data = _recover_units(data, table[col_name])
            axes.append(ObservationGroupAxis(col_name, data, fmt='values'))
            # format will be reviewed in a further step

        # detect range variables and eventually merge columns
        for i_col in range(len(axes)):
            try:
                split_name_min = axes[i_col].name.rsplit("_", 1)
                split_name_max = axes[i_col + 1].name.rsplit("_", 1)

                do_split = (
                    split_name_min[-1] == 'MIN' and
                    split_name_max[-1] == 'MAX' and
                    split_name_min[0] == split_name_max[0]
                )
                if do_split:
                    min_values = axes[i_col].bins
                    max_values = axes[i_col + 1].bins
                    edges = np.unique(np.append(min_values, max_values))
                    # recover units
                    edges = _recover_units(edges, min_values)

                    axes[i_col] = ObservationGroupAxis(split_name_min[0], edges, fmt='edges')
                    axes.pop(i_col + 1)  # remove next entry on the list
            except:
                pass

        return axes

    @classmethod
    def read(cls, filename):
        """Read observation group definitions from ECSV file.

        Using `~astropy.table.Table` and `~astropy.io.ascii`.

        Parameters
        ----------
        filename : str
            Name of the file.

        Returns
        -------
        obs_groups : `~gammapy.data.ObservationGroups`
            Observation groups object.
        """
        table = ascii.read(filename)
        axes = ObservationGroups.table_to_axes(table)
        return cls(axes=axes)

    def write(self, outfile, overwrite=False):
        """Write observation group definitions to ECSV file.

        Using `~astropy.table.Table` and `~astropy.io.ascii`.

        Parameters
        ----------
        outfile : str
            Name of the file.
        overwrite : bool, optional
            Flag to control file overwriting.
        """
        # there is no overwrite option in `~astropy.io.ascii`
        if not Path(outfile).is_file() or overwrite:
            ascii.write(self.obs_groups_table, outfile,
                        format='ecsv', fast_writer=False)

    @property
    def info(self):
        """Info string (str)."""
        s = ''
        # loop over observation axes
        for i_axis in range(len(self.axes)):
            s += self.axes[i_axis].info
            if i_axis < len(self.axes) - 1:
                s += '\n'
        return s

    def info_group(self, group_id):
        """Group info string.

        Parameters
        ----------
        group_id : int
            ID of the group to gather info on.

        Returns
        -------
        s : str
            Group info string.
        """
        s = 'group {}:'.format(group_id)
        # find group row in obs groups table
        group_ids = self.obs_groups_table['GROUP_ID'].data
        group_index = np.where(group_ids == group_id)
        row = group_index[0][0]
        # loop over observation axes
        for i_axis in range(len(self.axes)):
            if i_axis != 0:
                s += ','

            s += ' ' + self.axes[i_axis].name + ' = '

            if self.axes[i_axis].fmt == 'edges':
                s += '['
                s += str(self.obs_groups_table[self.axes[i_axis].name + '_MIN'][row])
                s += ', '
                s += str(self.obs_groups_table[self.axes[i_axis].name + '_MAX'][row])
                s += ')'
            elif self.axes[i_axis].fmt == 'values':
                s += str(self.obs_groups_table[self.axes[i_axis].name][row])

            s += ' ' + str(self.axes[i_axis].bins.unit)

        return s

    def apply(self, obs_table):
        """Group observations in a list according to the defined groups.

        The method returns the same observation table with an extra
        column in the 1st position indicating the group ID of each
        observation.

        The algorithm expects the same format (naming and variable
        definition range) for both the grouping axis definition and
        the corresponding variable in the table. For instance, if the
        azimuth axis binning is defined as ``AZ`` with bin edges
        ``[-90, 90, 270]`` (North and South bins), the input obs table
        should have an azimuth column defined as ``AZ`` and wrapped
        at ``270 deg``. This can easily be done by calling:

        >>> obs_table['AZ'] = Angle(obs_table['AZ']).wrap_at(Angle(270., 'deg'))

        Parameters
        ----------
        obs_table : `~gammapy.data.ObservationTable`
            Observation list to group.

        Returns
        -------
        obs_table_grouped : `~gammapy.data.ObservationTable`
            Grouped observation list.
        """
        # read the obs groups table row by row (i.e. 1 group at
        # a time) and lookup the range/value for each parameter
        n_axes = len(self.axes)
        list_obs_table_grouped = []
        for i_row in range(self.n_groups):
            i_group = self.obs_groups_table['GROUP_ID'][i_row]
            # loop over obs group axes to find out the names and formats
            # of the parameters to define the selection criteria
            obs_table_selected = obs_table
            for i_axis in range(n_axes):
                name = self.axes[i_axis].name
                fmt = self.axes[i_axis].fmt

                if fmt == 'edges':
                    min_value = _recover_units(self.obs_groups_table[name + '_MIN'][i_row],
                                               self.obs_groups_table[name + '_MIN'])
                    max_value = _recover_units(self.obs_groups_table[name + '_MAX'][i_row],
                                               self.obs_groups_table[name + '_MAX'])
                elif fmt == 'values':
                    min_value = _recover_units(self.obs_groups_table[name][i_row],
                                               self.obs_groups_table[name])
                    max_value = min_value
                # apply selection to the table
                selection = dict(type='par_box', variable=name,
                                 value_range=(min_value, max_value))
                obs_table_selected = obs_table_selected.select_observations(selection)

            # define group and fill in list of grouped observation tables
            group_id_data = i_group * np.ones(len(obs_table_selected), dtype=np.int)
            obs_table_selected.add_column(Column(name='GROUP_ID', data=group_id_data),
                                          index=0)
            list_obs_table_grouped.append(obs_table_selected)

        # stack all groups
        obs_table_grouped = vstack(list_obs_table_grouped)

        return obs_table_grouped

    def get_group_of_observations(self, obs_table, group,
                                  inverted=False, apply_grouping=False):
        """Select the runs corresponding to a particular group.

        If the inverted flag is activated, the selection is applied to
        exclude the indicated group and keep all others.

        Parameters
        ----------
        obs_table : `~gammapy.data.ObservationTable`
            Observation list to select from.
        group : int
            Group ID to select.
        inverted : bool, optional
            Invert selection: exclude the indicated group and keep the rest.
        apply_grouping : bool, optional
            Flag to indicate if the observation grouping should take place.

        Returns
        -------
        obs_table_group : `~gammapy.data.ObservationTable`
            Observation list of a specific group.
        """
        if apply_grouping:
            obs_table = self.apply(obs_table)

        selection = dict(type='par_box', variable='GROUP_ID',
                         value_range=(group, group), inverted=inverted)
        return obs_table.select_observations(selection)


class ObservationGroupAxis(object):
    """Observation group axis.

    Class to define an axis along which to define bins for creating observation groups.

    Two kinds of axis are supported, depending on the value of the ``fmt`` parameter:

    - ``fmt='edges'`` for continuous axes (e.g. altitude angle)
    - ``fmt='values'`` for discrete axes (e.g. number of telescopes)

    In both cases, both, dimensionless and
    `~astropy.units.Quantity`-like parameter axes are supported.

    See also :ref:`obs_observation_grouping`.

    Parameters
    ----------
    name : str
        Name of the parameter to bin.
    bins : int, float or `~astropy.units.Quantity`-like
        Array of values or bin edges, depending on the ``fmt`` parameter.
    fmt : {'edges', 'values'}
        Format of binning

    Examples
    --------

    Examples how to create ObservationGroupAxis objects:

    .. code-block:: python

        zenith = Angle([0, 30, 40, 50], 'deg')
        zenith_axis = ObservationGroupAxis('ALT', alt, fmt='edges')

        ntels = [3, 4]
        ntels_axis = ObservationGroupAxis('N_TELS', ntels, fmt='values')
    """

    def __init__(self, name, bins, fmt):
        if fmt not in ['edges', 'values']:
            raise ValueError("Invalid fmt: {}.".format(fmt))

        self.name = name
        self.bins = np.asanyarray(bins)
        self.fmt = fmt

    @property
    def n_bins(self):
        """Number of bins (int)."""
        if self.fmt == 'edges':
            return len(self.bins) - 1
        elif self.fmt == 'values':
            return len(self.bins)

    def get_bin(self, bin_id):
        """Get bin (int, float or `~astropy.units.Quantity`-like).

        Value or tuple of bin edges (depending on the ``fmt`` parameter)
        for the specified bin.

        Parameters
        ----------
        bin_id : int
            ID of the bin to retrieve.

        Returns
        -------
        bin : int, float or `~astropy.units.Quantity`-like
            Value or tuple of bin edges, depending on the ``fmt`` parameter.
        """
        if self.fmt == 'edges':
            return self.bins[bin_id], self.bins[bin_id + 1]
        elif self.fmt == 'values':
            return self.bins[bin_id]

    @property
    def get_bins(self):
        """List of bins (int, float or `~astropy.units.Quantity`-like).

        List of bin edges or values (depending on the ``fmt`` parameter)
        for all bins.
        """
        bins = []
        for i_bin in np.arange(self.n_bins):
            bins.append(self.get_bin(i_bin))

        return bins

    @classmethod
    def from_column(cls, col):
        """Import from astropy column.

        Parameters
        ----------
        col : `~astropy.table.Column`
            Column with the axis info.
        """
        return cls(
            name=col.name,
            bins=col,
            format=col.meta['axis_format']
        )

    def to_column(self):
        """Convert to astropy column.

        Returns
        -------
        col : `~astropy.table.Column`
            Column with the axis info.
         """
        col = Column(data=self.bins, name=self.name)
        col.meta['axis_format'] = self.fmt
        return col

    @property
    def info(self):
        """Info string (str)."""
        s = "{} {} {}".format(self.name, self.fmt, self.bins)
        return s


def _recover_units(array, as_units):
    """Utility function to recover units.

    After some numpy operations, `~astropy.units.Quantity`-like objects
    loose their units. This function shoul recover them.

    Parameters
    ----------
    array : `~numpy.ndarray` or `~astropy.units.Quantity`-like
        Array without units.
    as_units : int, float or `~astropy.units.Quantity`-like
        Structure to imitate the units.

    Returns
    -------
    array : int, float or `~astropy.units.Quantity`-like
        Array with units.
    """
    # TODO: this seems wrong to do stuff like this. Review and maybe remove?
    try:
        return Quantity(np.array(array), as_units.unit)
    except:
        # return unmodified
        return array

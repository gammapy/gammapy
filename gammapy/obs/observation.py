# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os.path
import logging
log = logging.getLogger(__name__)
import numpy as np
from astropy.table import Table, Column, vstack
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.io import ascii
from ..time import time_ref_from_dict, time_relative_to_ref
from ..catalog import select_sky_box, select_sky_circle

__all__ = [
    # 'Observation',
    'ObservationTable',
    'ObservationGroups',
    'ObservationGroupAxis',
]


class Observation(object):

    """Observation.

    An observation is a.k.a. run.
    TODO: not clear if this class is useful.

    Parameters
    ----------
    TODO
    """

    def __init__(self, GLON, GLAT, livetime=1800,
                 eff_area=1e12, background=0):
        self.GLON = GLON
        self.GLAT = GLAT
        self.livetime = livetime

    def wcs_header(self, system='FOV'):
        """Create a WCS FITS header for an per-run image.

        The image is centered on the run position in one of these systems:
        FOV, Galactic, Equatorial
        """
        raise NotImplementedError


class ObservationTable(Table):

    """Observation table (a.k.a. run list).

    This is an `~astropy.table.Table` sub-class, with a few
    convenience methods. The format of the observation table
    is described in :ref:`dataformats_observation_lists`.
    """

    @property
    def summary(self):
        """Info string (str)"""
        ss = 'Observation table:\n'
        obs_name = self.meta['OBSERVATORY_NAME']
        ss += 'Observatory name: {}\n'.format(obs_name)
        ss += 'Number of observations: {}\n'.format(len(self))
        ontime = Quantity(self['TIME_OBSERVATION'].sum(),
	                      self['TIME_OBSERVATION'].unit)
        ss += 'Total observation time: {}\n'.format(ontime)
        livetime = Quantity(self['TIME_LIVE'].sum(), self['TIME_LIVE'].unit)
        ss += 'Total live time: {}\n'.format(livetime)
        dtf = 100. * (1 - livetime / ontime)
        ss += 'Average dead time fraction: {:5.2f}%\n'.format(dtf)
        time_ref = time_ref_from_dict(self.meta)
        time_ref_unit = time_ref_from_dict(self.meta).format
        ss += 'Time reference: {} {}'.format(time_ref, time_ref_unit)
        return ss

    def select_linspace_subset(self, num):
        """Select subset of observations.

        This is mostly useful for testing, if you want to make
        the analysis run faster.

        TODO: implement more methods to subset and split observation lists
        as well as functions to summarise differences between
        observation lists and e.g. select the common subset.

        Parameters
        ----------
        num : int
            Number of samples to select.

        Returns
        -------
        table : `ObservationTable`
            Subset observation table (a copy).
        """
        indices = np.linspace(start=0, stop=len(self), num=num, endpoint=False)
        # Round down to nearest integer
        indices = indices.astype('int')
        return self[indices]

    def select_range(self, selection_variable, value_range, inverted=False):
        """Make an observation table, applying some selection.

        Generic function to apply a 1D box selection (min, max) to a
        table on any variable that is in the observation table and can
        be casted into a `~astropy.units.Quantity` object.

        If the range length is 0 (min = max), the selection is applied
        to the exact value indicated by the min value. This is useful
        for selection of exact values, for instance in discrete
        variables like the number of telescopes.

        If the inverted flag is activated, the selection is applied to
        keep all elements outside the selected range.

        Parameters
        ----------
        selection_variable : str
            Name of variable to apply a cut (it should exist on the table).
        value_range : `~astropy.units.Quantity`-like
            Allowed range of values (min, max). The type should be
            consistent with the selection_variable.
        inverted : bool, optional
            Invert selection: keep all entries outside the (min, max) range.

        Returns
        -------
        obs_table : `~gammapy.obs.ObservationTable`
            Observation table after selection.
        """
        obs_table = self

        # check that variable exists on the table
        if selection_variable not in obs_table.keys():
            raise KeyError('Key not present in table: {}'.format(selection_variable))

        # read values into a quantity in case units have to be taken into account
        value = Quantity(obs_table[selection_variable])

        # build and apply mask
        mask = (value_range[0] <= value) & (value < value_range[1])
        if np.allclose(value_range[0], value_range[1]):
            mask = value_range[0] == value
        if inverted:
            mask = np.invert(mask)
        obs_table = obs_table[mask]
        return obs_table

    def select_time_range(self, selection_variable, time_range, inverted=False):
        """Make an observation table, applying a time selection.

        Apply a 1D box selection (min, max) to a
        table on any time variable that is in the observation table.
        It supports both fomats: absolute times in
        `~astropy.time.Time` variables and [MET]_.

        If the inverted flag is activated, the selection is applied to
        keep all elements outside the selected range.

        Parameters
        ----------
        selection_variable : str
            Name of variable to apply a cut (it should exist on the table).
        time_range : `~astropy.time.Time`
            Allowed time range (min, max).
        inverted : bool, optional
            Invert selection: keep all entries outside the (min, max) range.

        Returns
        -------
        obs_table : `~gammapy.obs.ObservationTable`
            Observation table after selection.
        """
        obs_table = self

        # check that variable exists on the table
        if selection_variable not in obs_table.keys():
            raise KeyError('Key not present in table: {}'.format(selection_variable))

        if obs_table.meta['TIME_FORMAT'] == 'absolute':
            # read times into a Time object
            time = Time(obs_table[selection_variable])
        else:
            # transform time to MET
            time_range = time_relative_to_ref(time_range, obs_table.meta)
            # read values into a quantity in case units have to be taken into account
            time = Quantity(obs_table[selection_variable])

        # build and apply mask
        mask = (time_range[0] <= time) & (time < time_range[1])
        if inverted:
            mask = np.invert(mask)
        obs_table = obs_table[mask]
        return obs_table

    def select_observations(self, selection=None):
        """Make an observation table, applying some selection.

        There are 3 main kinds of selection criteria, according to the
        value of the **type** keyword in the **selection** dictionary:

        - sky regions (boxes or circles)

        - time intervals (min, max)

        - intervals (min, max) on any other parameter present in the
          observation table, that can be casted into an
          `~astropy.units.Quantity` object

        Allowed selection criteria are interpreted using the following
        keywords in the **selection** dictionary:

        - **type**: ``sky_box``, ``sky_circle``, ``time_box``, ``par_box``

            - ``sky_box`` and ``sky_circle`` are 2D selection criteria acting
              on sky coordinates

                - ``sky_box`` is a squared region delimited by the **lon** and
                  **lat** keywords: both tuples of format (min, max); uses
                  `~gammapy.catalog.select_sky_box`

                - ``sky_circle`` is a circular region centered in the coordinate
                  marked by the **lon** and **lat** keywords, and radius **radius**;
                  uses `~gammapy.catalog.select_sky_circle`

              in each case, the coordinate system can be specified by the **frame**
              keyword (built-in Astropy coordinate frames are supported, e.g.
              ``icrs`` or ``galactic``); an aditional border can be defined using
              the **border** keyword

            - ``time_box`` is a 1D selection criterion acting on the observation
              time (**TIME_START** and **TIME_STOP**); the interval is set via the
              **time_range** keyword; uses
              `~gammapy.obs.ObservationTable.select_time_range`

            - ``par_box`` is a 1D selection criterion acting on any
              parameter defined in the observation table that can be casted
              into an `~astropy.units.Quantity` object; the parameter name
              and interval can be specified using the keywords **variable** and
              **value_range** respectively; min = max selects exact
              values of the parameter; uses
              `~gammapy.obs.ObservationTable.select_range`

        In all cases, the selection can be inverted by activating the
        **inverted** flag, in which case, the selection is applied to keep all
        elements outside the selected range.

        A few examples of selection criteria are given below and more can be
        found in the tests in `test_select_observations`.

        Parameters
        ----------
        selection : dict
            Dictionary with a few keywords for applying selection cuts.

        Returns
        -------
        obs_table : `~gammapy.obs.ObservationTable`
            Observation table after selection.

        Examples
        --------
        >>> selection = dict(type='sky_box', frame='icrs',
        ...                  lon=Angle([150, 300], 'degree'),
        ...                  lat=Angle([-50, 0], 'degree'),
        ...                  border=Angle(2, 'degree'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='sky_circle', frame='galactic',
        ...                  lon=Angle(0, 'degree'),
        ...                  lat=Angle(0, 'degree'),
        ...                  radius=Angle(5, 'degree'),
        ...                  border=Angle(2, 'degree'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='time_box',
        ...                  time_range=Time(['2012-01-01T01:00:00', '2012-01-01T02:00:00'],
        ...                                  format='isot', scale='utc'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='ALT',
        ...                  value_range=Angle([60., 70.], 'degree'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='OBS_ID',
        ...                  value_range=[2, 5])
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='N_TELS',
        ...                  value_range=[4, 4])
        >>> selected_obs_table = obs_table.select_observations(selection)
        """
        obs_table = self

        if selection:
            selection_type = selection['type']

            if 'inverted' not in selection.keys():
                selection['inverted'] = False

            if selection_type == 'sky_circle':
                lon = selection['lon']
                lat = selection['lat']
                radius = selection['radius'] + selection['border']
                obs_table = select_sky_circle(obs_table,
                                              lon_cen=lon, lat_cen=lat,
                                              radius=radius,
                                              frame=selection['frame'],
                                              inverted=selection['inverted'])

            elif selection_type == 'sky_box':
                lon = selection['lon']
                lat = selection['lat']
                border = selection['border']
                lon = Angle([lon[0] - border, lon[1] + border])
                lat = Angle([lat[0] - border, lat[1] + border])
                obs_table = select_sky_box(obs_table,
                                           lon_lim=lon, lat_lim=lat,
                                           frame=selection['frame'],
                                           inverted=selection['inverted'])

            elif selection_type == 'time_box':
                # apply twice the mask: to TIME_START and TIME_STOP
                obs_table = obs_table.select_time_range('TIME_START',
                                                        selection['time_range'],
                                                        selection['inverted'])
                obs_table = obs_table.select_time_range('TIME_STOP',
                                                        selection['time_range'],
                                                        selection['inverted'])

            elif selection_type == 'par_box':
                obs_table = obs_table.select_range(selection['variable'],
                                                   selection['value_range'],
                                                   selection['inverted'])

            else:
                raise ValueError('Invalid selection type: {}'.format(selection_type))

        return obs_table


def recover_units(array, as_units):
    """Utility functin to recover units.

    After some numpy operations, `~astropy.units.Quantity`-like objects
    loose their units. This function shoul recover them.

    TODO: extend to Time arrays.

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
    try:
        return Quantity(np.array(array), as_units.unit)
    except:
        # return unmodified
        return array


class ObservationGroups(object):

    """Observation groups.

    Class to define observation groups useful for organizing observation
    lists into groups of observations with similar properties. The
    properties and their binning are specified via
    `~gammapy.obs.ObservationGroupAxis` objects.

    The class takes as input a list of `~gammapy.obs.ObservationGroupAxis`
    objects and defines 1 group for each possible combination of the
    bins defined in all axes. The groups are identified by a unique
    ``GROUP_ID`` int value.

    The definitions of the groups are internally  stored as a
    `~astropy.table.Table` object, the
    `~gammapy.obs.ObservationGroups.obs_groups_table` member.

    The axis parameters should be either dimensionless or castable
    into `~astropy.units.Quantity` objects.

    For details on the grouping of observations in a list, please
    refer to the `~gammapy.obs.ObservationGroups.group_observation_table`
    method.

    TODO: show a grouped obs list and a table of obs groups in the high-level docs
    (and a list of axes)!!!!
    (do it in the "future" page for the "future" inline command tool
    for obs groups!!!)

    Parameters
    ----------
    obs_group_axes : `~gammapy.obs.ObservationGroupAxis`
        List of observation group axes.

    Examples
    --------
    Define an observation grouping:

    .. code:: python

        alt = Angle([0, 30, 60, 90], 'degree')
        az = Angle([-90, 90, 270], 'degree')
        ntels = np.array([3, 4])
        list_obs_group_axis = [ObservationGroupAxis('ALT', alt, 'bin_edges'),
                               ObservationGroupAxis('AZ', az, 'bin_edges'),
                               ObservationGroupAxis('N_TELS', ntels, 'bin_values')]
        obs_group = ObservationGroups(list_obs_group_axis)

    Print the observation group table (group definitions):

    >>> print(obs_group.obs_groups_table)

    Print the observation group axes:

    >>> print(obs_group.info)

    Group the observations of an observation list and print it:

    >>> obs_table_grouped = obs_group.group_observation_table(obs_table)
    >>> print(obs_table_grouped)
    """

    obs_groups_table = Table()

    def __init__(self, obs_group_axes):
        self.obs_group_axes = obs_group_axes
        if len(self.obs_groups_table) == 0:
            self.define_groups(self.axes_to_table(self.obs_group_axes))

    def define_groups(self, table):
        """Define observation groups for a given table of bins.

        Define one group for each possible combination of the
        observation group axis bins, defined as rows in the
        input table.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table with observation group axis bins combined.
        """
        if len(self.obs_groups_table.columns) is not 0:
            raise RuntimeError(
                "Catched attempt to overwrite existing obs groups table.")

        # define number of groups
        n_groups = 1
        # loop over observation axes
        for i_axis in np.arange(len(self.obs_group_axes)):
            n_groups *= self.obs_group_axes[i_axis].n_bins

        if len(table) is not n_groups:
            raise ValueError("Invalid table length. Got {0}, expected {1}".format(
                len(table), n_groups))

        # fill table, with first the obs group IDs, then the axis columns
        self.obs_groups_table = table
        self.obs_groups_table.add_column(Column(name='GROUP_ID',
                                                data=np.arange(n_groups)),
                                         index=0)

    @property
    def n_groups(self):
        """Number of groups (int)"""
        return len(self.obs_groups_table)

    @property
    def list_of_groups(self):
        """List of groups (`~numpy.ndarray`)"""
        return self.obs_groups_table['GROUP_ID'].data

    def axes_to_table(self, axes):
        """Fill the observation group axes into a table.

        Define one row for each possible combination of the
        observation group axis bins. Each row will represent
        an observation group.

        Parameters
        ----------
        axes : `~gammapy.obs.ObservationGroupAxis`
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
        for i_axis in np.arange(len(axes)):
            if axes[i_axis].format == 'bin_values':
                column_data_min.append(axes[i_axis].bins)
                column_data_max.append(axes[i_axis].bins)
            elif axes[i_axis].format == 'bin_edges':
                column_data_min.append(axes[i_axis].bins[:-1])
                column_data_max.append(axes[i_axis].bins[1:])

        # define grids of column data
        ndim = len(axes)
        s0 = (1,)*ndim
        expanding_arrays = [x.reshape(s0[:i] + (-1,) + s0[i + 1::])
                            for i, x in enumerate(column_data_min)]
        column_data_expanded_min = np.broadcast_arrays(*expanding_arrays)
        expanding_arrays = [x.reshape(s0[:i] + (-1,) + s0[i + 1::])
                            for i, x in enumerate(column_data_max)]
        column_data_expanded_max = np.broadcast_arrays(*expanding_arrays)

        # recover units
        for i_dim in np.arange(ndim):
            column_data_expanded_min[i_dim] = recover_units(column_data_expanded_min[i_dim],
                                                            column_data_min[i_dim])
            column_data_expanded_max[i_dim] = recover_units(column_data_expanded_max[i_dim],
                                                            column_data_max[i_dim])

        # define table columns
        columns = []
        for i_axis in np.arange(len(axes)):
            if axes[i_axis].format == 'bin_values':
                columns.append(Column(data=column_data_expanded_min[i_axis].flatten(),
                                      name=axes[i_axis].name))
            elif axes[i_axis].format == 'bin_edges':
                columns.append(Column(data=column_data_expanded_min[i_axis].flatten(),
                                      name=axes[i_axis].name + "_MIN"))
                columns.append(Column(data=column_data_expanded_max[i_axis].flatten(),
                                      name=axes[i_axis].name + "_MAX"))

        # fill table
        table = Table()
        for i, col in enumerate(columns):
            table.add_column(col)

        return table

    def table_to_axes(self, table):
        """Define observation group axis list from a table.

        Interpret the combinations of bins from a table of groups
        in order to define the corresponding observation group axes.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table containing the observation group definitions.

        Returns
        -------
        axes : `~gammapy.obs.ObservationGroupAxis`
            List of observation group axes.
        """
        # subset table: remove obs groups column
        if table.colnames[0] == 'GROUP_ID':
            table = table[table.colnames[1:]]

        axes = []
        for i_col, col_name in enumerate(table.columns):
            data = np.unique(table[col_name].data)
            # recover units
            data = recover_units(data, table[col_name])
            axes.append(ObservationGroupAxis(col_name, data,
                                             'bin_values'))
            # format will be reviewed in a further step
            # TODO: maybe it's better to store/read the parameter
            #       format in/from the table header?!!!

        # detect range variables and eventually merge columns
        for i_col in np.arange(len(axes)):
            try:
                split_name_min = axes[i_col].name.rsplit("_", 1)
                split_name_max = axes[i_col + 1].name.rsplit("_", 1)
                if (split_name_min[-1] == 'MIN'
                    and split_name_max[-1] == 'MAX'
                    and split_name_min[0] == split_name_max[0]):
                    min_values = axes[i_col].bins
                    max_values = axes[i_col + 1].bins
                    edges = np.unique(np.append(min_values, max_values))
                    # recover units
                    edges = recover_units(edges, min_values)

                    axes[i_col] = ObservationGroupAxis(split_name_min[0], edges,
                                                       'bin_edges')
                    axes.pop(i_col + 1) # remove next entry on the list
            except:
                pass

        return axes

    @classmethod
    def read(cls, filename):
        """
        Read observation group definitions from ECSV file.

        Using `~astropy.table.Table` and `~astropy.io.ascii`.

        Parameters
        ----------
        filename : str
            Name of the file.

        Returns
        -------
        obs_groups : `~gammapy.obs.ObservationGroups`
            Observation groups object.
        """
        cls.obs_groups_table = ascii.read(filename)
        return cls(obs_group_axes=cls.table_to_axes(cls, cls.obs_groups_table))

    def write(self, outfile, overwrite=False):
        """
        Write observation group definitions to ECSV file.

        Using `~astropy.table.Table` and `~astropy.io.ascii`.

        Parameters
        ----------
        outfile : str
            Name of the file.
        overwrite : bool, optional
            Flag to control file overwriting.
        """
        # there is no overwrite option in `~astropy.io.ascii`
        if not os.path.isfile(outfile) or overwrite:
            ascii.write(self.obs_groups_table, outfile,
                        format='ecsv', fast_writer=False)

    @property
    def info(self):
        """Info string (str)"""
        s = ''
        # loop over observation axes
        for i_axis in np.arange(len(self.obs_group_axes)):
            s += self.obs_group_axes[i_axis].info
            if i_axis < len(self.obs_group_axes) - 1:
                s += '\n'
        return s

    def print_axes(self):
        """Print axes info using the logging module."""
        #print(self.info)
        log.info(self.info)

    def print_groups(self):
        """Print groups info using the logging module."""
        #print(self.obs_groups_table)
        log.info(print(self.obs_groups_table))

    def group_observation_table(self, obs_table):
        """
        Group observations in a list according to the defined groups.

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

        >>> obs_table['AZ'] = Angle(obs_table['AZ']).wrap_at(Angle(270., 'degree'))

        Parameters
        ----------
        obs_table : `~gammapy.obs.ObservationTable`
            Observation list to group.

        Returns
        -------
        obs_table_grouped : `~gammapy.obs.ObservationTable`
            Grouped observation list.
        """
        if 'GROUP_ID' in obs_table.colnames:
            raise KeyError(
                "Catched attempt to overwrite existing grouping in the table.")

        # read the obs groups table row by row (i.e. 1 group at
        # a time) and lookup the range/value for each parameter
        n_axes = len(self.obs_group_axes)
        list_obs_table_grouped = []
        for i_row in np.arange(self.n_groups):
            i_group = self.obs_groups_table['GROUP_ID'][i_row]
            # loop over obs group axes to find out the names and formats
            # of the parameters to define the selection criteria
            obs_table_selected = obs_table
            for i_axis in np.arange(n_axes):
                name = self.obs_group_axes[i_axis].name
                format = self.obs_group_axes[i_axis].format

                if format == 'bin_edges':
                    min_value = recover_units(self.obs_groups_table[name + '_MIN'][i_row],
                                              self.obs_groups_table[name + '_MIN'])
                    max_value = recover_units(self.obs_groups_table[name + '_MAX'][i_row],
                                              self.obs_groups_table[name + '_MAX'])
                elif format == 'bin_values':
                   min_value = recover_units(self.obs_groups_table[name][i_row],
                                              self.obs_groups_table[name])
                   max_value = min_value
                # apply selection to the table
                selection = dict(type='par_box', variable=name,
                                 value_range=(min_value, max_value))
                obs_table_selected = obs_table_selected.select_observations(selection)
            # define group and fill in list of grouped observation tables
            group_id_data = i_group*np.ones(len(obs_table_selected), dtype=np.int)
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
        obs_table : `~gammapy.obs.ObservationTable`
            Observation list to select from.
        group : int
            Group ID to select.
        inverted : bool, optional
            Invert selection: exclude the indicated group and keep the rest.
        apply_grouping : bool, optional
            Flag to indicate if the observation grouping should take place.

        Returns
        -------
        obs_table_group : `~gammapy.obs.ObservationTable`
            Observation list of a specific group.
        """
        if apply_grouping:
            obs_table = self.group_observation_table(obs_table)

        selection = dict(type='par_box', variable='GROUP_ID',
                         value_range=(group, group), inverted=inverted)
        return obs_table.select_observations(selection)


class ObservationGroupAxis(object):

    """Observation group axis.

    Class to define an axis along which to define observation groups.
    Two kinds of axis are supported, depending on the value of the
    **format** parameter:

    - **format**: ``bin_edges``, ``bin_values``

        - ``bin_edges`` defines a continuous axis (eg. altitude angle)

        - ``bin_values`` defines a discrete axis (eg. number of telescopes)

    In both cases, both, dimensionless and
    `~astropy.units.Quantity`-like parameter axes are supported.

    Parameters
    ----------
    name : str
        Name of the parameter to bin.
    bins : int, float or `~astropy.units.Quantity`-like
        Array of values or bin edges, depending on the **format** parameter.
    format : str
        Format of binning specified: ``bin_edges``, ``bin_values``.

    Examples
    --------
    Create a few axes:

    .. code:: python

        alt = Angle([0, 30, 60, 90], 'degree')
        alt_obs_group_axis = ObservationGroupAxis('ALT', alt, 'bin_edges')
        az = Angle([-90, 90, 270], 'degree')
        az_obs_group_axis = ObservationGroupAxis('AZ', az, 'bin_edges')
        ntels = np.array([3, 4])
        ntels_obs_group_axis = ObservationGroupAxis('N_TELS', ntels, 'bin_values')
    """

    def __init__(self, name, bins, format):
        if format not in ['bin_edges', 'bin_values']:
            raise ValueError("Invalid bin format {}.".format(self.format))
        self.name = name
        self.bins = bins
        self.format = format

    @property
    def n_bins(self):
        """Number of bins (int)"""
        if self.format == 'bin_edges':
            return len(self.bins) - 1
        elif self.format == 'bin_values':
            return len(self.bins)

    def get_bin(self, bin_id):
        """Get bin (int, float or `~astropy.units.Quantity`-like)

        Value or tuple of bin edges (depending on the **format** parameter)
        for the specified bin.

        Parameters
        ----------
        bin_id : int
            ID of the bin to retrieve.

        Returns
        -------
        bin : int, float or `~astropy.units.Quantity`-like
            Value or tuple of bin edges, depending on the **format** parameter.
        """
        if self.format == 'bin_edges':
            return (self.bins[bin_id], self.bins[bin_id + 1])
        elif self.format == 'bin_values':
            return self.bins[bin_id]

    @property
    def get_bins(self):
        """List of bins (int, float or `~astropy.units.Quantity`-like)

        List of bin edges or values (depending on the **format** parameter)
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
        return cls(name=col.name,
                   bins=col,
                   format=col.meta['axis_format'])

    def to_column(self):
        """Convert to astropy column.

        Returns
        -------
        col : `~astropy.table.Column`
            Column with the axis info.
         """
        col = Column(data=self.bins, name=self.name)
        col.meta['axis_format'] = self.format
        return col

    @property
    def info(self):
        """Info string (str)"""
        return ("{0} {1} {2}".format(self.name, self.format, self.bins))

    def print(self):
        """Print axis info using the logging module."""
        #print(self.info)
        log.info(self.info)

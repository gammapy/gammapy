# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.time import Time
from ..time import time_ref_from_dict, time_relative_to_ref
from ..catalog import select_sky_box

__all__ = [
    # 'Observation',
    'ObservationTable',
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


    def summary(self):
        ss = 'Observation table:\n'
        obs_name = self.meta['OBSERVATORY_NAME']
        ss += 'Observatory name: {}\n'.format(obs_name)
        ss += 'Number of observations: {}\n'.format(len(self))
        ontime = Quantity(self['TIME_OBSERVATION'].sum(), self['TIME_OBSERVATION'].unit)
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


    def filter_observations_var_range(self, selection_variable,
                                      value_min, value_max, inverted):
        """Make an observation table, applying some selection.

        Generic function to apply a 1D box selection (min, max) to a
        table on any variable that is in the observation table. In
        addition, support has been added for `zenith` anlge, computed
        as `(90 deg - altitude)`.
        If the inverted flag is activated, the filter is applied to
        keep all elements outside the selected range.

        Parameters
        ----------
        selection_variable : string
            name of variable to apply a cut (it should exist on the table)
        value_min : TBD
            minimum value; type should be consistent with selection_variable
        value_max : TBD
            maximum value; type should be consistent with selection_variable
        inverted : bool
            invert selection: keep all entries outside the (min, max) range

        Returns
        -------
        obs_table : `~gammapy.obs.ObservationTable`
            observation table with observations passing the selection
        """
        obs_table = self

        # check that variable exists on the table
        special_values = ['zenith']
        if selection_variable not in obs_table.keys():
            # some values are still recognized
            if selection_variable not in special_values:
                raise KeyError('Key not present in table: {}'.format(selection_variable))

        if selection_variable == 'zenith':
            # transform zenith to altitude
            selection_variable = 'ALT'
            zenith_min = value_min
            zenith_max = value_max
            value_min = Angle(90., 'degree') - zenith_max
            value_max = Angle(90., 'degree') - zenith_min
            # read values into a quantity in case units have to be taken into account
            value = Quantity(obs_table[selection_variable])
        elif selection_variable in ['TIME_START', 'TIME_STOP']:
            if obs_table.meta['TIME_FORMAT'] == 'absolute':
                # read values into a Time object
                value = Time(obs_table[selection_variable])
            else:
                # transform time to MET
                value_min = time_relative_to_ref(value_min, obs_table.meta)
                value_max = time_relative_to_ref(value_max, obs_table.meta)
                # read values into a quantity in case units have to be taken into account
                value = Quantity(obs_table[selection_variable])
        else:
            # read values into a quantity in case units have to be taken into account
            value = Quantity(obs_table[selection_variable])

        # build and apply mask
        if not inverted:
            mask = (value_min < value) & (value < value_max)
        else:
            mask = (value_min >= value) | (value >= value_max)
        obs_table = obs_table[mask]
        return obs_table


    def filter_observations(self, selection=None):
        """Make an observation table, applying some selection.

        Allowed selection criteria are interpreted using the following
        keywords in the `selection` dictionary:

            - `shape`: ``box``, ``circle``, ``sky_box``, ``sky_circle``

                - ``box`` and ``circle`` are 1D selection criteria acting on any
                  variable defined in the observation table, specified using the
                  `variable` keyword

                    - `box` is an interval delimited by the `value_min` and
                      `value_max` parameters

                    - `circle` is a centered interval defined by the `center`
                      and `radius` parameters

                - ``sky_box`` and ``sky_circle`` are 2D selection criteria acting
                  on sky coordinates, similar to ``box`` and ``circle``
                  TODO: finish implementing and documenting sky_box and sky_circle!!!

            In all cases, the selection can be inverted by activating the
            `inverted` flag, in which case, the filter is applied to keep all
            elements outside the selected range.

        A few examples of selection criteria can be found in the tests in
        `test_filter_observations`.
        TODO: is there a way to insert a non-hard-coded link to the tests?!!!

        Parameters
        ----------
        selection : `~dict`
            dictionary with a few keywords for applying selection cuts

        Returns
        -------
        obs_table : `~gammapy.obs.ObservationTable`
            observation table with observations passing the selection

        Examples
        --------
        >>> selection = dict(shape='sky_box', frame='galactic',
        ...                  lon=(-100, 50), lat=(-5, 5), border=2)
        >>> filtered_obs_table = obs_table.filter_observations(selection)
        TODO: update example (or remove it, since we have an exhaustive doc?) !!!!!!!!!!!!!!!!!!!!!!
        """
        obs_table = self

        if selection:
            selection_region_shape = selection['shape']
            if 'variable' in selection.keys():
                selection_variable = selection['variable']
            inverted = False
            if 'inverted' in selection.keys():
                inverted = selection['inverted']

            if selection_region_shape == 'box':
                value_min = selection['value_min']
                value_max = selection['value_max']
                if selection_variable == 'time':
                    # apply twice the mask: to TIME_START and TIME_STOP
                    obs_table = obs_table.filter_observations_var_range('TIME_START',
                                                                        value_min,
                                                                        value_max,
                                                                        inverted)
                    obs_table = obs_table.filter_observations_var_range('TIME_STOP',
                                                                        value_min,
                                                                        value_max,
                                                                        inverted)
                else:
                    # regular case
                    obs_table = obs_table.filter_observations_var_range(selection_variable,
                                                                        value_min,
                                                                        value_max,
                                                                        inverted)
            elif selection_region_shape == 'circle':
                value_min = selection['center'] - selection['radius']
                value_max = selection['center'] + selection['radius']
                if selection_variable == 'time':
                    # apply twice the mask: to TIME_START and TIME_STOP
                    obs_table = obs_table.filter_observations_var_range('TIME_START',
                                                                        value_min,
                                                                        value_max,
                                                                        inverted)
                    obs_table = obs_table.filter_observations_var_range('TIME_STOP',
                                                                        value_min,
                                                                        value_max,
                                                                        inverted)
                else:
                    # regular case
                    obs_table = obs_table.filter_observations_var_range(selection_variable,
                                                                        value_min,
                                                                        value_max,
                                                                        inverted)
            elif selection_region_shape == 'sky_box':
                lon = selection['lon']
                lat = selection['lat']
                border = selection['border']
                lon = Angle([lon[0] - border, lon[1] + border], 'deg')
                lat = Angle([lat[0] - border, lat[1] + border], 'deg')
                # print(lon, lat)
                obs_table = select_sky_box(obs_table,
                                           lon_lim=lon, lat_lim=lat,
                                           frame=selection['frame'])
            elif selection_region_shape == 'sky_circle':
                raise NotImplemented
            else:
                raise ValueError('Invalid selection type: {}'.format(selection_region_shape))

        return obs_table

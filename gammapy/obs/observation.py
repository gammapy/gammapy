# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.time import Time
from ..time import time_ref_from_dict, time_relative_to_ref
from ..catalog import select_sky_box, select_sky_circle

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

    def select_range(self, selection_variable, value_range, inverted=False):
        """Make an observation table, applying some selection.

        Generic function to apply a 1D box selection (min, max) to a
        table on any variable that is in the observation table and can
        be casted into a `~astropy.units.Quantity` object.

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
        value of the `type` keyword in the `selection` dictionary:

        - sky regions (boxes or circles)

        - time intervals (min, max)

        - intervals (min, max) on any other parameter present in the
          observation table, that can be casted into an
          `~astropy.units.Quantity` object

        Allowed selection criteria are interpreted using the following
        keywords in the `selection` dictionary:

        - `type`: ``sky_box``, ``sky_circle``, ``'time_box``, ``par_box``

            - ``sky_box`` and ``sky_circle`` are 2D selection criteria acting
              on sky coordinates

                - ``sky_box`` is a squared region delimited by the `lon` and
                  `lat` keywords: both tuples of format (min, max); uses
                  `~gammapy.catalog.select_sky_box`

                - ``sky_circle`` is a circular region centered in the coordinate
                  marked by the `lon` and `lat` keywords, and radius `radius`;
                  uses `~gammapy.catalog.select_sky_circle`

              in each case, the coordinate system can be specified by the `frame`
              keyword (built-in Astropy coordinate frames are supported, e.g.
              \'icrs\' or \'galactic\'); an aditional border can be defined using
              the `border` keyword

            - ``time_box`` is a 1D selection criterion acting on the observation
              time (`TIME_START` and `TIME_STOP`); the interval is set via the
              `time_range` keyword; uses
              `~gammapy.obs.ObservationTable.select_time_range`

            - ``par_box`` is a 1D selection criterion acting on any
              parameter defined in the observation table that can be casted
              into an `~astropy.units.Quantity` object; the parameter name
              and interval can be specified using the keywords 'variable' and
              'value_range' respectively; uses
              `~gammapy.obs.ObservationTable.select_range`

        In all cases, the selection can be inverted by activating the
        `inverted` flag, in which case, the selection is applied to keep all
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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astropy.utils import lazyproperty
from ..utils.scripts import make_path
from ..utils.time import time_relative_to_ref

__all__ = [
    'ObservationTable',
]


class ObservationTable(Table):
    """Observation table.

    Data format specification: :ref:`gadf:obs-index`
    """

    @classmethod
    def read(cls, filename, **kwargs):
        """Read an observation table from file.

        Parameters
        ----------
        filename : `~gammapy.extern.pathlib.Path`, str
            Filename
        """
        filename = make_path(filename)
        return super(ObservationTable, cls).read(str(filename), **kwargs)

    @property
    def pointing_radec(self):
        """Pointing positions as ICRS (`~astropy.coordinates.SkyCoord`)"""
        return SkyCoord(self['RA_PNT'], self['DEC_PNT'], unit='deg', frame='icrs')

    @property
    def pointing_galactic(self):
        """Pointing positions as Galactic (`~astropy.coordinates.SkyCoord`)"""
        return SkyCoord(self['GLON_PNT'], self['GLAT_PNT'], unit='deg', frame='galactic')

    @lazyproperty
    def _index_dict(self):
        """Dict containing row index for all obs ids"""
        # TODO: Switch to http://docs.astropy.org/en/latest/table/indexing.html once it is more stable
        temp = (zip(self['OBS_ID'], np.arange(len(self))))
        return dict(temp)

    def get_obs_idx(self, obs_id):
        """Get row index for given ``obs_id``.

        Raises KeyError if observation is not available.

        Parameters
        ----------
        obs_id : int, list
            observation ids

        Returns
        -------
        idx : list
            indices corresponding to obs_id
        """
        idx = [self._index_dict[key] for key in np.atleast_1d(obs_id)]
        return idx

    def select_obs_id(self, obs_id):
        """Get `~gammapy.data.ObservationTable` containing only ``obs_id``.

        Raises KeyError if observation is not available.

        Parameters
        ----------
        obs_id: int, list
            observation ids
        """
        return self[self.get_obs_idx(obs_id)]

    def summary(self):
        """Info string (str)"""
        obs_name = self.meta.get('OBSERVATORY_NAME', 'N/A')

        return '\n'.join([
            'Observation table:',
            'Observatory name: {!r}'.format(obs_name),
            'Number of observations: {}'.format(len(self)),
            # TODO: clean this up. Make those properties?
            # ontime = Quantity(self['ONTIME'].sum(), self['ONTIME'].unit)
            #
            # ss += 'Total observation time: {}\n'.format(ontime)
            # livetime = Quantity(self['LIVETIME'].sum(), self['LIVETIME'].unit)
            # ss += 'Total live time: {}\n'.format(livetime)
            # dtf = 100. * (1 - livetime / ontime)
            # ss += 'Average dead time fraction: {:5.2f}%\n'.format(dtf)
            # time_ref = time_ref_from_dict(self.meta)
            # time_ref_unit = time_ref_from_dict(self.meta).format
            # ss += 'Time reference: {} {}'.format(time_ref, time_ref_unit)
        ])

    def select_linspace_subset(self, num):
        """Select subset of observations.

        This is mostly useful for testing, if you want to make
        the analysis run faster.

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
        obs_table : `~gammapy.data.ObservationTable`
            Observation table after selection.
        """
        value_range = Quantity(value_range)

        # read values into a quantity in case units have to be taken into account
        value = Quantity(self[selection_variable])

        mask = (value_range[0] <= value) & (value < value_range[1])

        if np.allclose(value_range[0].value, value_range[1].value):
            mask = (value_range[0] == value)

        if inverted:
            mask = np.invert(mask)

        return self[mask]

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
        obs_table : `~gammapy.data.ObservationTable`
            Observation table after selection.
        """
        if self.meta['TIME_FORMAT'] == 'absolute':
            # read times into a Time object
            time = Time(self[selection_variable])
        else:
            # transform time to MET
            time_range = time_relative_to_ref(time_range, self.meta)
            # read values into a quantity in case units have to be taken into account
            time = Quantity(self[selection_variable])

        mask = (time_range[0] <= time) & (time < time_range[1])

        if inverted:
            mask = np.invert(mask)

        return self[mask]

    def select_observations(self, selection=None):
        """Select subset of observations.

        Returns a new observation table representing the subset.

        There are 3 main kinds of selection criteria, according to the
        value of the **type** keyword in the **selection** dictionary:

        - sky regions (boxes or circles)

        - time intervals (min, max)

        - intervals (min, max) on any other parameter present in the
          observation table, that can be casted into an
          `~astropy.units.Quantity` object

        Allowed selection criteria are interpreted using the following
        keywords in the **selection** dictionary under the **type** key.

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
          start time (**TSTART**); the interval is set via the
          **time_range** keyword; uses
          `~gammapy.data.ObservationTable.select_time_range`

        - ``par_box`` is a 1D selection criterion acting on any
          parameter defined in the observation table that can be casted
          into an `~astropy.units.Quantity` object; the parameter name
          and interval can be specified using the keywords **variable** and
          **value_range** respectively; min = max selects exact
          values of the parameter; uses
          `~gammapy.data.ObservationTable.select_range`

        In all cases, the selection can be inverted by activating the
        **inverted** flag, in which case, the selection is applied to keep all
        elements outside the selected range.

        A few examples of selection criteria are given below.

        Parameters
        ----------
        selection : dict
            Dictionary with a few keywords for applying selection cuts.

        Returns
        -------
        obs_table : `~gammapy.data.ObservationTable`
            Observation table after selection.

        Examples
        --------
        >>> selection = dict(type='sky_box', frame='icrs',
        ...                  lon=Angle([150, 300], 'deg'),
        ...                  lat=Angle([-50, 0], 'deg'),
        ...                  border=Angle(2, 'deg'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='sky_circle', frame='galactic',
        ...                  lon=Angle(0, 'deg'),
        ...                  lat=Angle(0, 'deg'),
        ...                  radius=Angle(5, 'deg'),
        ...                  border=Angle(2, 'deg'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='time_box',
        ...                  time_range=Time(['2012-01-01T01:00:00', '2012-01-01T02:00:00']))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='ALT',
        ...                  value_range=Angle([60., 70.], 'deg'))
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='OBS_ID',
        ...                  value_range=[2, 5])
        >>> selected_obs_table = obs_table.select_observations(selection)

        >>> selection = dict(type='par_box', variable='N_TELS',
        ...                  value_range=[4, 4])
        >>> selected_obs_table = obs_table.select_observations(selection)
        """
        from ..catalog import select_sky_box, select_sky_circle

        if 'inverted' not in selection.keys():
            selection['inverted'] = False

        if selection['type'] == 'sky_circle':
            lon = selection['lon']
            lat = selection['lat']
            radius = selection['radius'] + selection['border']
            return select_sky_circle(
                self, lon_cen=lon, lat_cen=lat, radius=radius,
                frame=selection['frame'], inverted=selection['inverted']
            )

        elif selection['type'] == 'sky_box':
            lon = selection['lon']
            lat = selection['lat']
            border = selection['border']
            lon = Angle([lon[0] - border, lon[1] + border])
            lat = Angle([lat[0] - border, lat[1] + border])
            return select_sky_box(
                self, lon_lim=lon, lat_lim=lat,
                frame=selection['frame'], inverted=selection['inverted']
            )

        elif selection['type'] == 'time_box':
            return self.select_time_range(
                'TSTART', selection['time_range'], selection['inverted']
            )

        elif selection['type'] == 'par_box':
            return self.select_range(
                selection['variable'], selection['value_range'], selection['inverted']
            )

        else:
            raise ValueError('Invalid selection type: {}'.format(selection['type']))

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle
from ..time import time_ref_from_dict
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


    def filter_observations_var_range(self, selection_variable, value_min, value_max, inverted):
        """Make an observation table, applying some selection.

        Generic function to apply a 1D box selection [min, max) to a table on any variable.
        TODO: not working for zenith angle!!!!
              and if it would: [ , ) selection is applied ( , ], right?!!!
              (this [ , ) thing is also tricky for circle selections!!!) -> remove the "="?? (i.e. do a ( , ) selection?!!!))!!!!

        Parameters
        ----------
        selection_variable : string
            name of variable to apply a cut (it should exist on the table)
        value_min : ???!!!
            minimum value
        value_max : ???!!!
            maximum value
        inverted : bool
            invert selection: keep all entries outside the [min, max) range

        Returns
        -------
        obs_table : `~gammapy.obs.ObservationTable`
            observation table with observations passing the selection        """
        obs_table = self
        # check that variable exists on the table
        if selection_variable not in obs_table.keys():
            raise KeyError('Key not present in table: {}'.format(selection_variable))
        value = obs_table[selection_variable]
        if not inverted:
            mask = (value_min <= value) & (value < value_max)
        else:
            mask = (value_min > value) | (value >= value_max)
        obs_table = obs_table[mask]
        return obs_table

        
    def filter_observations(self, selection=None):
        """Make an observation table, applying some selection.

        TODO: implement a more flexible scheme to make box cuts
        on any fields (including e.g. OBSID or TIME
        Not sure what a simple yet powerful method to implement this is!?

        Parameters
        ----------
        selection : `~dict`
            dictionary with a few keywords for applying selection cuts
            TODO: add doc with allowed selection cuts!!! and link here!!!

        Returns
        -------
        obs_table : `~gammapy.obs.ObservationTable`
            observation table with observations passing the selection

        Examples
        --------
        >>> selection = dict(shape='box', frame='galactic',
        ...                  lon=(-100, 50), lat=(-5, 5), border=2)
        >>> filtered_obs_table = obs_table.filter_observations(selection)
        """
        obs_table = self

        #TODO: search code for criteria implemented!!!
        # in datastore, in event lists, ...?
        # gammapy/catalog/utils.py select_sky_box
        # check that I don't break anything because of missing tests in existing code!!! (i.e. in https://github.com/mapazarr/hess-host-analyses/blob/master/hgps_survey_map/hgps_survey_map.py#L62)
        #TOOD: implement script that provides a run list and outputs a filtered run list!!!

# test if it works with ra dec (is in that case long = RA, lat = dec? (or does it cleverly transform coordinates and makes selection?)
# do a more user-friendly way of giving ra dec (without long lat)!!!

        if selection:
            selection_region_shape = selection['shape']
            if 'variable' in selection.keys():
                selection_variable = selection['variable']
            inverted = False
            if 'inverted' in selection.keys():
                inverted = selection['inverted']

            if selection_region_shape == 'box':
                value_min = selection['min']
                value_max = selection['max']
                obs_table = obs_table.filter_observations_var_range(selection_variable, value_min, value_max, inverted)
            elif selection_region_shape == 'circle':
                value_min = selection['center'] - selection['radius']
                value_max = selection['center'] + selection['radius']
                obs_table = obs_table.filter_observations_var_range(selection_variable, value_min, value_max, inverted)
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

#TODO: should I take already Angle objects as parameters in the selection???!!!!!!!!!!!!!!!


##box (min, max)
##circle (center, radius)
##
##1D
##2D -> sky_box, sky_circle
##
##inverted?

        return obs_table

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.io import fits
from ..background import Cube,CubeBackgroundModel

__all__ = ['make_bg_cube_model',
           ]

def make_bg_cube_model(observation_table, fits_path, DEBUG):
    """Create a bg model from an observation table.

    Produce a background cube model using the data from an observation list.
    Steps:

    1. define binning
    2. fill events and livetime correction in cubes
    3. fill events in bg cube
    4. smooth
    5. correct for livetime and bin volume
    6. set 0 level

    TODO: review steps!!!

    Parameters
    ----------
    observation_table : `~gammapy.obs.ObservationTable`
        Observation list to use for the histogramming.
    fits_path : str
        Path to the data files.
    DEBUG : int
        Debug level.

    Returns
    -------
    events_cube : `~gammapy.background.Cube`
        Cube containing the events.
    livetime_cube : `~gammapy.background.Cube`
        Cube containing the livetime.
    bg_cube : `~gammapy.background.Cube`
        Cube background model.
    """

    # DEBUG: 0: no output, 1: output, 2: NOTHING, 3: more verbose
    # TODO: remove the DEBUG variable, when the logger works!!!
    # TODO: I need to pass the logger or at least the log level!!!!
    #       and remove the DEBUG option!!!
    #       Look how the DataStore does it (when importing a file).

    ##################
    # define binning #
    ##################

    energy_edges, dety_edges, detx_edges = CubeBackgroundModel.define_cube_binning(len(observation_table), DEBUG)


    ####################################################
    # create empty cubes: events, livetime, background #
    ####################################################

    empty_cube_data = np.zeros((len(energy_edges) - 1,
                                len(dety_edges) - 1,
                                len(detx_edges) - 1))

    if DEBUG:
        print("cube shape", empty_cube_data.shape)

    events_cube = Cube(detx_bins=detx_edges,
                       dety_bins=dety_edges,
                       energy_bins=energy_edges,
                       data=Quantity(empty_cube_data, '')) # counts

    livetime_cube = Cube(detx_bins=detx_edges,
                         dety_bins=dety_edges,
                         energy_bins=energy_edges,
                         data=Quantity(empty_cube_data, 'second'))

    bg_cube = Cube(detx_bins=detx_edges,
                   dety_bins=dety_edges,
                   energy_bins=energy_edges,
                   data=Quantity(empty_cube_data, '1 / (s TeV sr)'))


    ############################
    # fill events and livetime #
    ############################

    # TODO: filter out possible sources in the data
    #       for now, the observation table should not contain any
    #       run at or near an existing source

    events_cube, livetime_cube = CubeBackgroundModel.fill_events(observation_table, fits_path,
                                             events_cube, livetime_cube,
                                             DEBUG)


    ################
    # fill bg cube #
    ################

    bg_cube.data = events_cube.data


    ##########
    # smooth #
    ##########

    bg_cube = CubeBackgroundModel.smooth(bg_cube, events_cube.data.sum())


    #######################################################
    # correct for livetime and bin volume and set 0 level #
    #######################################################

    bg_cube.data /= livetime_cube.data
    bg_cube = CubeBackgroundModel.divide_bin_volume(bg_cube)
    bg_cube = CubeBackgroundModel.set_zero_level(bg_cube)


    ######################
    # return the 3 cubes #
    ######################

    return events_cube, livetime_cube, bg_cube

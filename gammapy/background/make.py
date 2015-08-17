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

    The background cube model contains 3 cubes: events, livetime and background.
    The latter contains the bg cube model.

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
    bg_cube_model : `~gammapy.background.CubeBackgroundModel`
        Cube background model.
    """

    # DEBUG: 0: no output, 1: output, 2: NOTHING, 3: more verbose
    # TODO: remove the DEBUG variable, when the logger works!!!
    # TODO: I need to pass the logger or at least the log level!!!!
    #       and remove the DEBUG option!!!
    #       Look how the DataStore does it (when importing a file).

    ####################################################################
    # define cube background model object with binning and empty cubes #
    ####################################################################

    bg_cube_model = CubeBackgroundModel.define_cube_binning(len(observation_table), DEBUG)

    if DEBUG:
        print("cube shape", bg_cube_model.background_cube.data.shape)


    ############################
    # fill events and livetime #
    ############################

    # TODO: filter out possible sources in the data
    #       for now, the observation table should not contain any
    #       run at or near an existing source

    bg_cube_model.fill_events(observation_table, fits_path, DEBUG)


    ################
    # fill bg cube #
    ################

    bg_cube_model.background_cube.data = bg_cube_model.events_cube.data


    ##########
    # smooth #
    ##########

    bg_cube_model.smooth()


    #######################################################
    # correct for livetime and bin volume and set 0 level #
    #######################################################

    bg_cube_model.background_cube.data /= bg_cube_model.livetime_cube.data
    bg_cube_model.background_cube.divide_bin_volume()
    bg_cube_model.background_cube.set_zero_level()


    ############################
    # return the bg cube model #
    ############################

    return bg_cube_model

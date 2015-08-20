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


def make_bg_cube_model(observation_table, fits_path, a_la_michi=False):
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
    a_la_michi : bool, optional
        Flag to activate Michael Mayer's bg cube production method.
        Correct for livetime before applying the smoothing.
        Use units of *1 / (MeV sr s)* for the bg rate.

    Returns
    -------
    bg_cube_model : `~gammapy.background.CubeBackgroundModel`
        Cube background model.
    """
    ####################################################################
    # define cube background model object with binning and empty cubes #
    ####################################################################

    bg_cube_model = CubeBackgroundModel.define_cube_binning(observation_table,
                                                            fits_path,
                                                            do_not_fill=False,
                                                            a_la_michi=a_la_michi)


    ############################
    # fill events and livetime #
    ############################

    # TODO: filter out (mask) possible sources in the data
    #       for now, the observation table should not contain any
    #       run at or near an existing source

    bg_cube_model.fill_events(observation_table, fits_path)


    ################
    # fill bg cube #
    ################

    bg_cube_model.background_cube.data = bg_cube_model.events_cube.data.copy()
    if a_la_michi:
        # correct for livetime before smoothing
        bg_cube_model.background_cube.data /= bg_cube_model.livetime_cube.data


    ##########
    # smooth #
    ##########

    bg_cube_model.smooth()


    #######################################################
    # correct for livetime and bin volume and set 0 level #
    #######################################################

    if not a_la_michi:
        # correct for livetime after smoothing
        bg_cube_model.background_cube.data /= bg_cube_model.livetime_cube.data
    bg_cube_model.background_cube.divide_bin_volume()
    if a_la_michi:
        # use units of 1 / (MeV sr s) for the bg rate
        bg_rate = bg_cube_model.background_cube.data.to('1 / (MeV sr s)')
        bg_cube_model.background_cube.data = bg_rate
    bg_cube_model.background_cube.set_zero_level()


    ############################
    # return the bg cube model #
    ############################

    return bg_cube_model

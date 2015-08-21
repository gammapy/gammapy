# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.io import fits
from ..background import Cube, CubeBackgroundModel

__all__ = ['make_bg_cube_model',
           ]


def make_bg_cube_model(observation_table, fits_path, method='default'):
    """Create a bg model from an observation table.

    Produce a background cube model using the data from an observation list.
    Steps:

    1. define binning
    2. fill events and livetime correction in cubes
    3. fill events in bg cube
    4. smooth
    5. correct for livetime and bin volume
    6. set 0 level

    The steps are slightly altered in case a different method as the
    *default* one is used. In the *michi* method:

    * Correction for livetime applied before the smoothing.
    * Units of *1 / (MeV sr s)* for the bg rate are used.

    The background cube model contains 3 cubes: events, livetime and background.
    The latter contains the bg cube model.

    Parameters
    ----------
    observation_table : `~gammapy.obs.ObservationTable`
        Observation list to use for the histogramming.
    fits_path : str
        Path to the data files.
    method : {'default', 'michi'}, optional
        Bg cube model calculation method to apply.

    Returns
    -------
    bg_cube_model : `~gammapy.background.CubeBackgroundModel`
        Cube background model.
    """
    if method=='default':
        bg_cube_model = CubeBackgroundModel.define_cube_binning(observation_table,
                                                                fits_path,
                                                                do_not_fill=False,
                                                                method=method)
        bg_cube_model.fill_events(observation_table, fits_path)
        # TODO: filter out (mask) possible sources in the data
        #       for now, the observation table should not contain any
        #       run at or near an existing source
        bg_cube_model.background_cube.data = bg_cube_model.counts_cube.data.copy()
        bg_cube_model.smooth()
        bg_cube_model.background_cube.data /= bg_cube_model.livetime_cube.data
        bg_cube_model.background_cube.divide_bin_volume()
        bg_cube_model.background_cube.set_zero_level()

        return bg_cube_model

    elif method == 'michi':
        bg_cube_model = CubeBackgroundModel.define_cube_binning(observation_table,
                                                                fits_path,
                                                                do_not_fill=False,
                                                                method=method)
        bg_cube_model.fill_events(observation_table, fits_path)
        # TODO: filter out (mask) possible sources in the data
        #       for now, the observation table should not contain any
        #       run at or near an existing source
        bg_cube_model.background_cube.data = bg_cube_model.counts_cube.data.copy()
        bg_cube_model.background_cube.data /= bg_cube_model.livetime_cube.data
        bg_cube_model.smooth()
        # correct for livetime after smoothing
        bg_cube_model.background_cube.divide_bin_volume()
        # use units of 1 / (MeV sr s) for the bg rate
        bg_rate = bg_cube_model.background_cube.data.to('1 / (MeV sr s)')
        bg_cube_model.background_cube.data = bg_rate
        bg_cube_model.background_cube.set_zero_level()

        return bg_cube_model

    else:
        raise ValueError("Invalid method {}.".format(method))

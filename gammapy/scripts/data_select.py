# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.time import Time
from ..utils.scripts import get_parser, set_up_logging_from_args
from ..data import ObservationTable

__all__ = ['data_select_main']

log = logging.getLogger(__name__)


def data_select_main(args=None):
    """Main function for argument parsing."""
    parser = get_parser(data_select)
    parser.add_argument('infile', type=str,
                        help='Input observation table file name (fits format)')
    parser.add_argument('outfile', nargs='?', type=str,
                        default=None,
                        help='Output observation table file name '
                             '(default: None, will print the result on screen)')
    parser.add_argument('--x', type=float, default=None,
                        help='x coordinate (deg)')
    parser.add_argument('--y', type=float, default=None,
                        help='y coordinate (deg)')
    parser.add_argument('--r', type=float, default=None,
                        help='circle radius (deg)')
    parser.add_argument('--dx', type=float, default=None,
                        help='box semi-length x coordinate (deg)')
    parser.add_argument('--dy', type=float, default=None,
                        help='box semi-length y coordinate (deg)')
    parser.add_argument('--system', type=str,
                        help='Coordinate system '
                             '(built-in Astropy coordinate frames are supported, '
                             'e.g. \'icrs\' or \'galactic\')')
    parser.add_argument('--t_start', type=str, default=None,
                        help='UTC start time (string: yyyy-mm-ddThh:mm:ss.sssssssss)')
    parser.add_argument('--t_stop', type=str, default=None,
                        help='UTC end time (string: yyyy-mm-ddThh:mm:ss.sssssssss)')
    parser.add_argument('--par_name', type=str, default=None,
                        help='Parameter name (included in the input)')
    parser.add_argument('--par_min', type=float, default=None,
                        help='Parameter min value (units as in obs table)')
    parser.add_argument('--par_max', type=float, default=None,
                        help='Parameter max value (units as in obs table)')
    parser.add_argument('--invert', type=bool, default=False,
                        help='If true, invert the selection')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)

    set_up_logging_from_args(args)

    data_select(**vars(args))


def data_select(infile,
                outfile,
                x,
                y,
                r,
                dx,
                dy,
                system,
                t_start,
                t_stop,
                par_name,
                par_min,
                par_max,
                invert,
                overwrite):
    """Select a subset of observations from a given observation list.

    This inline tool selects observations from a an input observation
    table fits file and prints the output on screen or saves it to an
    output fits file.

    For a detailed description of the options, please use the help
    option of this tool by calling:

    .. code-block:: bash

        gammapy-obs-select -h

    In order to test the examples below, the test observation list
    file located in the ``gammapy-extra`` repository `test_observation_table.fits`_
    can be used as input observation list.

    .. _test_observation_table.fits: https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/obs/test_observation_table.fits

    More information is available at :ref:`obs_find_observations`.

    Examples
    --------

    .. code-block:: bash

        gammapy-obs-select -h
        gammapy-obs-select test_observation_table.fits
        gammapy-obs-select test_observation_table.fits out.test_observation_table.fits --overwrite
        gammapy-obs-select test_observation_table.fits --x 130 --y -40 --r 50 --system 'icrs'
        gammapy-obs-select test_observation_table.fits --x 0 --y 0 --r 50 --system 'galactic'
        gammapy-obs-select test_observation_table.fits --x 225 --y -25 --dx 75 --dy 25 --system 'icrs'
        gammapy-obs-select test_observation_table.fits --x -25 --y 0 --dx 75 --dy 25 --system 'galactic'
        gammapy-obs-select test_observation_table.fits --t_start '2012-01-01T00:00:00' --t_stop '2014-01-01T00:00:00'
        gammapy-obs-select test_observation_table.fits --par_name 'OBS_ID' --par_min 2 --par_max 6
        gammapy-obs-select test_observation_table.fits --par_name 'ALT' --par_min 60 --par_max 70
        gammapy-obs-select test_observation_table.fits --par_name 'N_TELS' --par_min 4 --par_max 4
    """
    # open (fits) file and read the observation table
    try:
        observation_table = ObservationTable.read(infile)
    except IOError:
        log.error('File not found: {}'.format(infile))
        exit(-1)

    # sky circle selection
    do_sky_circle_selection = np.array([(x != None), (y != None),
                                        (r != None), (system != None)])
    if do_sky_circle_selection.all():
        log.debug("Applying sky circle selection.")
        # cast x, y, r into Angle objects
        lon_cen = Angle(x, 'deg')
        lat_cen = Angle(y, 'deg')
        radius = Angle(r, 'deg')
        selection = dict(type='sky_circle', frame=system,
                         lon=lon_cen, lat=lat_cen,
                         radius=radius, border=Angle(0., 'deg'),
                         inverted=invert)
        observation_table = observation_table.select_observations(selection)
    else:
        if r is not None:
            raise ValueError("Could not apply sky circle selection.")

    # sky box selection
    do_sky_box_selection = np.array(
        [(x is not None), (y is not None), (dx is not None), (dy is not None), (system is not None)]
    )
    if do_sky_box_selection.all():
        log.debug("Applying sky box selection.")
        # convert x, y, dx, dy to ranges and cast into Angle objects
        lon_range = Angle([x - dx, x + dx], 'deg')
        lat_range = Angle([y - dy, y + dy], 'deg')
        selection = dict(type='sky_box', frame=system,
                         lon=(lon_range[0], lon_range[1]),
                         lat=(lat_range[0], lat_range[1]),
                         border=Angle(0., 'deg'),
                         inverted=invert)
        observation_table = observation_table.select_observations(selection)
    else:
        if (dx is not None) or (dy is not None):
            raise ValueError("Could not apply sky box selection.")

    # time box selection
    do_time_box_selection = np.array([(t_start != None), (t_stop != None)])
    if do_time_box_selection.all():
        log.debug("Applying time box selection.")
        # convert min, max to range and cast into Time object
        t_range = Time([t_start, t_stop])
        selection = dict(type='time_box', time_range=t_range, inverted=invert)
        observation_table = observation_table.select_observations(selection)
    else:
        if do_time_box_selection.any():
            raise ValueError("Could not apply time box selection.")

    # generic parameter box selection
    do_par_box_selection = np.array([(par_name != None),
                                     (par_min != None), (par_max != None)])
    if do_par_box_selection.all():
        log.debug("Applying {} selection.".format(par_name))
        # convert min, max to range and cast into Quantity object with unit
        unit = observation_table[par_name].unit
        par_range = Quantity([par_min, par_max], unit)
        selection = dict(type='par_box', variable=par_name,
                         value_range=par_range, inverted=invert)
        observation_table = observation_table.select_observations(selection)
    else:
        if do_par_box_selection.any():
            raise ValueError("Could not apply parameter box selection.")

    if outfile is not None:
        observation_table.write(outfile, overwrite=overwrite)
    else:
        log.info("Filtered observation table")
        log.info(observation_table.meta)
        print(observation_table)

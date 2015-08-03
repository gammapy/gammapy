# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import argparse
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.time import Time
from ..utils.scripts import get_parser
from ..obs import ObservationTable

__all__ = ['find_obs']


def main(args=None):
    """Main function for argument parsing."""
    parser = get_parser(find_obs)
    parser.add_argument('infile', type=str,
                        help='Input obseravtion table file name (fits format)')
    parser.add_argument('outfile', nargs='?', type=str,
                        default=None,
                        help='Output obseravtion table file name (default: stdout)')
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
                        'e.g. \'icrs\' or \'galactic\'.)')
    parser.add_argument('--pix', action='store_true',
                        help='Input coordinates are pixels '
                        '(default is world coordinates)')
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
    args = parser.parse_args(args)
    find_obs(**vars(args))


def find_obs(infile,
             outfile,
             x,
             y,
             r,
             dx,
             dy,
             system,
             pix,
             t_start,
             t_stop,
             par_name,
             par_min,
             par_max,
             invert,
             overwrite):
    """Select a subset of observations from a given observation list.

    WARNING: this is still a PRELIMINARY version of the tool.

    I still have a few TODO's to work out, but the script does select
    observations froma an observation table fits file and prints the
    output on screen or saves it to a fits file.

    Stringdoc still missing; for now, please have a look at
    https://ms2.physik.hu-berlin.de/~mapaz/gammapy/docs/_build/html/obs/find_observations.html

    For testing, download this file from `~gammapy-extra`

    https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/obs/test_observation_table.fits
    
    And use the following commands in your shell terminal:

    .. code-block:: bash

        gammapy-find-obs -h
        gammapy-find-obs test_observation_table.fits
        gammapy-find-obs test_observation_table.fits out.test_observation_table.fits --overwrite
        gammapy-find-obs test_observation_table.fits --x 130 --y -40 --r 50 --system 'icrs'
        gammapy-find-obs test_observation_table.fits --x 0 --y 0 --r 50 --system 'galactic'
        gammapy-find-obs test_observation_table.fits --x 225 --y -25 --dx 75 --dy 25 --system 'icrs'
        gammapy-find-obs test_observation_table.fits --x -25 --y 0 --dx 75 --dy 25 --system 'galactic'
        gammapy-find-obs test_observation_table.fits --t_start '2012-01-01T00:00:00' --t_stop '2014-01-01T00:00:00'
        gammapy-find-obs test_observation_table.fits --par_name 'OBS_ID' --par_min 2 --par_max 6
        gammapy-find-obs test_observation_table.fits --par_name 'ALT' --par_min 60 --par_max 70

    TODO: explain (edit this docstring) + link to gammapy/docs/_build/html/obs/find_observations doc.

    TODO: update docs
    old: https://gammapy.readthedocs.org/en/latest/obs/findruns.html
    new: file:///home/mapaz/astropy/development_code/gammapy/docs/_build/html/obs/find_observations.html

    TODO: write tests for this command-line tool!!!
    """
    print("WARNING: this is still a PRELIMINARY version of the tool.")
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    if pix:
        raise NotImplementedError

    # open (fits) file and read the observation table
    observation_table = ObservationTable.read(infile)

    # sky circle selection
    do_sky_circle_selection = np.array([(x != None), (y != None),
                                        (r != None), (system != None)])
    if do_sky_circle_selection.all():
        print("Applying sky circle selection") # TODO: use logging!!!
        # cast x, y, r into Angle objects
        lon_cen = Angle(x, 'degree')
        lat_cen = Angle(y, 'degree')
        radius = Angle(r, 'degree')
        selection = dict(type='sky_circle', frame=system,
                         lon=lon_cen, lat=lat_cen,
                         radius=radius, border=Angle(0., 'degree'),
                         inverted=invert)
        observation_table = observation_table.select_observations(selection)
    else:
        if r is not None:
            raise ValueError("Could not apply sky circle selection.")

    # sky box selection
    do_sky_box_selection = np.array([(x != None), (y != None),
                                     (dx != None), (dy != None),
                                     (system != None)])
    if do_sky_box_selection.all():
        print("Applying sky box selection") # TODO: use logging!!!
        # convert x, y, dx, dy to ranges and cast into Angle objects
        lon_range = Angle([x - dx, x + dx], 'degree')
        lat_range = Angle([y - dy, y + dy], 'degree')
        selection = dict(type='sky_box', frame=system,
                         lon=(lon_range[0], lon_range[1]),
                         lat=(lat_range[0], lat_range[1]),
                         border=Angle(0., 'degree'),
                         inverted=invert)
        observation_table = observation_table.select_observations(selection)
    else:
        if (dx is not None) or (dy is not None):
            raise ValueError("Could not apply sky box selection.")

    # time box selection
    do_time_box_selection = np.array([(t_start != None), (t_stop != None)])
    if do_time_box_selection.all():
        print("Applying time box selection") # TODO: use logging!!!
        # cast min, max into Time objects
        t_start = Time(t_start, format='isot', scale='utc')
        t_stop = Time(t_stop, format='isot', scale='utc')
        selection = dict(type='time_box',
                         time_min=t_start, time_max=t_stop,
                         inverted=invert)
        observation_table = observation_table.select_observations(selection)
    else:
        if do_time_box_selection.any():
            raise ValueError("Could not apply time box selection.")

    # generic parameter box selection
    do_par_box_selection = np.array([(par_name != None),
                                     (par_min != None), (par_max != None)])
    if do_par_box_selection.all():
        print("Applying {} selection".format(par_name)) # TODO: use logging!!!
        # cast min, max into Quantity objects with units
        unit = observation_table[par_name].unit
        par_min = Quantity(par_min, unit)
        par_max = Quantity(par_max, unit)
        selection = dict(type='par_box', variable=par_name,
                         value_min=par_min, value_max=par_max,
                         inverted=invert)
        observation_table = observation_table.select_observations(selection)
    else:
        if do_par_box_selection.any():
            raise ValueError("Could not apply parameter box selection.")
    # TODO: allow multiple var selections!!! (read arrays/lists)

    if outfile is not None:
        observation_table.write(outfile, overwrite=overwrite)
    else:
        print(observation_table.meta)
        print(observation_table)
        # TODO: output to stdout!!!

    print("WARNING: this is still a PRELIMINARY version of the tool.")

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
log = logging.getLogger(__name__)
from ..utils.scripts import get_parser, set_up_logging_from_args

__all__ = ['reflected_regions']


def main(args=None):
    parser = get_parser(reflected_regions)
    parser.add_argument('--x_on', type=float,
                        help='On-region X position (deg)')
    parser.add_argument('--y_on', type=float,
                        help='On-region Y position (deg)')
    parser.add_argument('--r_on', type=float,
                        help='On-region radius (deg)')
    parser.add_argument('--x_fov', type=float,
                        help='FOV X position (deg)')
    parser.add_argument('--y_fov', type=float,
                        help='FOV Y position (deg)')
    parser.add_argument('--r_fov', type=float,
                        help='FOV radius (deg)')
    parser.add_argument('--exclusion', type=str,
                        help='Exclusion region mask FITS file name')
    parser.add_argument('--outfile', type=str,
                        help='Output file name (ds9 region format) '
                        '[default=%(default)s]')
    parser.add_argument('--min_on_distance', type=float, default=0.1,
                        help='Minimum distance to the on region (deg)')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)
    set_up_logging_from_args(args)
    reflected_regions(**vars(args))


def reflected_regions(x_on,
                      y_on,
                      r_on,
                      x_fov,
                      y_fov,
                      r_fov,
                      exclusion,
                      outfile,
                      min_on_distance):
    """Find off regions for a given on region and exclusion mask.

    TODO: explain a bit.
    """
    from astropy.io import fits
    from gammapy.background import ReflectedRegionMaker

    if exclusion:
        log.info('Reading {0}'.format(exclusion))
        exclusion = fits.open(exclusion)[0]
    else:
        # log.info('No exclusion mask used.')
        # TODO: make this work without exclusion mask
        log.error("Currently an exclusion mask is required")
        exit(-1)

    fov = dict(x=x_fov, y=y_fov, r=r_fov)
    rr_maker = ReflectedRegionMaker(exclusion=exclusion,
                                    fov=fov)
    source = dict(x_on=x_on, y_on=y_on, r_on=r_on)
    rr_maker.compute(**source)

    log.info('Writing {0}'.format(outfile))
    rr_maker.write_off_regions(outfile)

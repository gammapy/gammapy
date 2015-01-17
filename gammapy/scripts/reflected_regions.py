# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['reflected_regions']


def main(args=None):
    from gammapy.utils.scripts import argparse, GammapyFormatter
    description = reflected_regions.__doc__.split('\n')[0]
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=GammapyFormatter)
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
    args = parser.parse_args(args)
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
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    from astropy.io import fits
    from gammapy.background import ReflectedRegionMaker

    logging.info('Reading {0}'.format(exclusion))
    exclusion = fits.open(exclusion)[0]

    fov = dict(x=x_fov, y=y_fov, r=r_fov)
    rr_maker = ReflectedRegionMaker(exclusion=exclusion,
                                    fov=fov)
    source = dict(x_on=x_on, y_on=y_on, r_on=r_on)
    rr_maker.compute(**source)
    logging.info('Writing {0}'.format(outfile))
    rr_maker.write_off_regions(outfile)

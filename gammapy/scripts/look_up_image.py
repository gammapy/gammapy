# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['look_up_image']

# TODO: support coordinate parsing
# TODO: support multiple input images
# TODO: support reading lists of coordinates from FITS and CVS tables
# TODO: implement world coordinates option (`pix` option)


def main(args=None):
    from gammapy.utils.scripts import argparse, GammapyFormatter
    description = look_up_image.__doc__.split('\n')[0]
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=GammapyFormatter)
    parser.add_argument('infile', type=str,
                        help='Input FITS file name')
    parser.add_argument('x', type=float,
                        help='x coordinate (deg)')
    parser.add_argument('y', type=float,
                        help='y coordinate (deg)')
    # parser.add_argument('--pix', action='store_true',
    #                    help='Input coordinates are in pixels? (world coordinates if false)')
    args = parser.parse_args(args)
    look_up_image(**vars(args))


def look_up_image(infile,
                  x,
                  y,
                  # pix,
                  ):
    """Look up values in a map at given positions."""
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    from gammapy.utils.fits import get_hdu
    from gammapy.image import lookup

    logging.debug('Reading {0}'.format(infile))
    hdu = get_hdu(infile)

    value = lookup(hdu, x, y)
    print('Map value at position ({0}, {1}) is {2}'.format(x, y, value))

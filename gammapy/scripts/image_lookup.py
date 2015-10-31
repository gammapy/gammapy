# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..utils.scripts import get_parser

__all__ = ['image_lookup']

log = logging.getLogger(__name__)


# TODO: support coordinate parsing
# TODO: support multiple input images
# TODO: support reading lists of coordinates from FITS and CSV tables
# TODO: implement world coordinates option (`pix` option)


def image_lookup_main(args=None):
    parser = get_parser(image_lookup)
    parser.add_argument('infile', type=str,
                        help='Input FITS file name')
    parser.add_argument('x', type=float,
                        help='x coordinate (deg)')
    parser.add_argument('y', type=float,
                        help='y coordinate (deg)')
    # parser.add_argument('--pix', action='store_true',
    #                    help='Input coordinates are in pixels? (world coordinates if false)')
    args = parser.parse_args(args)
    image_lookup(**vars(args))


def image_lookup(infile,
                 x,
                 y,
                 # pix,
                 ):
    """Look up values in a map at given positions."""
    from gammapy.utils.fits import get_hdu
    from gammapy.image import lookup

    log.debug('Reading {0}'.format(infile))
    hdu = get_hdu(infile)

    value = lookup(hdu, x, y)
    print('Map value at position ({0}, {1}) is {2}'.format(x, y, value))

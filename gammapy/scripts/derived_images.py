# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
from ..utils.scripts import get_parser

__all__ = ['derived_images']

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(derived_images)
    parser.add_argument('infile', type=str,
                        help='Input FITS file name')
    parser.add_argument('outfile', type=str,
                        help='Output FITS file name')
    parser.add_argument('theta', type=float,
                        help='On-region correlation radius (deg)')
    parser.add_argument('--is_off_correlated', action='store_false',
                        help='Are the basic OFF maps correlated?')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    derived_images(**vars(args))


def derived_images(infile,
                   outfile,
                   theta,
                   is_off_correlated,
                   overwrite):
    """Make derived maps for a given set of basic maps.

    TODO: describe
    """
    from astropy.io import fits
    from gammapy.background import Maps

    log.info('Reading {0}'.format(infile))
    hdus = fits.open(infile)
    maps = Maps(hdus, theta=theta,
                is_off_correlated=is_off_correlated)
    log.info('Computing derived maps')
    maps.make_derived_maps()
    log.info('Writing {0}'.format(outfile))
    maps.writeto(outfile, clobber=overwrite)

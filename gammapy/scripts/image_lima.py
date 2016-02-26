# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click
click.disable_unicode_literals_warning = True

from astropy.io import fits
from astropy.convolution import Tophat2DKernel

from ..detect import compute_lima_map, compute_lima_on_off_map
from ..image.utils import dict_to_hdulist

__all__ = ['image_lima']

log = logging.getLogger(__name__)


def image_lima_main(args=None):
    parser = get_parser(image_derived)
    parser.add_argument('infile', type=str,
                        help='Input FITS file name')
    parser.add_argument('outfile', type=str,
                        help='Output FITS file name')
    parser.add_argument('theta', type=float,
                        help='On-region correlation radius (deg)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    image_derived(**vars(args))


def image_lima(infile, outfile, theta, overwrite):
    """
    Compute Li&Ma significance maps for a given set of input maps.

    TODO: describe
    """
    log.info('Reading {0}'.format(infile))
    hdus = fits.open(infile)

    # Convert theta to pix
    theta_pix = theta / hdus[0].header['CDELT2']

    kernel = Tophat2DKernel(theta_pix)
    result = compute_lima_map(hdus['counts'], hdu['background'],
                              hdus['exposure'], kernel)
    log.info('Computing derived maps')

    log.info('Writing {0}'.format(outfile))
    hdu_list = dict_to_hdulist(result, header)
    hdu_list.writeto(outfile, clobber=overwrite)

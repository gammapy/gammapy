# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
from ..utils.scripts import get_parser

__all__ = ['image_decompose_a_trous']

log = logging.getLogger(__name__)


# TODO: add option to copy over input file
# TODO: add option to select input HDU name or number


def main(args=None):
    parser = get_parser(image_decompose_a_trous)
    parser.add_argument('infile', type=str,
                        help='Input FITS file name')
    parser.add_argument('outfile', type=str,
                        help='Output FITS file name')
    parser.add_argument('n_levels', type=int,
                        help='Number of levels (wavelet scales)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    image_decompose_a_trous(**vars(args))


def image_decompose_a_trous(infile,
                            outfile,
                            n_levels,
                            overwrite):
    """Apply the a trous wavelet transform on a 2D image.

    This book contains an overview of methods:
    http://www.amazon.de/dp/3540330240

    Specifically I'd like to have an a trous transform.
    Here's a few references that look useful:
    * https://code.google.com/p/image-funcut/

    I'm also interested in these transforms:
    * http://scikit-image.org/docs/dev/api/skimage.transform.html#pyramid-expand
    * http://scikit-image.org/docs/dev/api/skimage.transform.html#pyramid-gaussian
    * http://scikit-image.org/docs/dev/api/skimage.transform.html#pyramid-laplacian
    * http://scikit-image.org/docs/dev/api/skimage.transform.html#pyramid-reduce
    """
    from astropy.io import fits
    from gammapy.image.utils import atrous_hdu

    log.info('Reading {0}'.format(infile))
    hdu = fits.open(infile)[0]
    atrous_hdus = atrous_hdu(hdu, n_levels=n_levels)

    log.info('Writing {0}'.format(outfile))
    atrous_hdus.writeto(outfile, clobber=overwrite)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..utils.scripts import get_parser

__all__ = ['image_model']

log = logging.getLogger(__name__)


def image_model_main(args=None):
    parser = get_parser(image_model)
    parser.add_argument('--exposure', type=str, default='exposure.fits',
                        help='Exposure FITS file name')
    parser.add_argument('--psf', type=str, default='psf.json',
                        help='PSF JSON file name')
    parser.add_argument('--sources', type=str, default='sources.json',
                        help='Sources JSON file name (contains start '
                             'values for fit of Gaussians)')
    parser.add_argument('--outfile', type=str, default='model.fits',
                        help='Output model image FITS file name')
    args = parser.parse_args(args)
    image_model(**vars(args))


def image_model(exposure,
                psf,
                sources,
                outfile):
    """Given a catalog of sources, simulate a flux image.

    Inputs:

    * Source list (JSON file)
    * PSF (JSON file)
    * Exposure image (FITS file)

    Outputs:

    * Source model flux image (FITS file)
    * Source model excess image (FITS file)
    """
    # raise NotImplementedError
    # from astropy.io import fits
    # from ..image import empty_image
    # from ..image import _to_image_bbox as to_image
    #
    # catalog = fits.open('test_catalog.fits')[1].data
    # image = empty_image(nxpix=600, nypix=600, binsz=0.02, xref=0, yref=0, dtype='float64')
    # to_image(catalog, image)
    #
    # log.info('Writing {}'.format(outfile))
    # image.writetofits(outfile)
    pass

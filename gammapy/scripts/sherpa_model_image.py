# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['sherpa_model_image']


def main(args=None):
    from gammapy.utils.scripts import argparse, GammapyFormatter
    description = sherpa_model_image.__doc__.split('\n')[0]
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=GammapyFormatter)
    parser.add_argument('--exposure', type=str, default='exposure.fits',
                        help='Exposure FITS file name')
    parser.add_argument('--psf', type=str, default='psf.json',
                        help='PSF JSON file name ')
    parser.add_argument('--sources', type=str, default='sources.json',
                        help='Sources JSON file name (contains start '
                        'values for fit of Gaussians)')
    parser.add_argument('--model_image', type=str, default='model.fits',
                        help='Output model image FITS file name')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    sherpa_model_image(**vars(args))


def sherpa_model_image(exposure,
                       psf,
                       sources,
                       model_image,
                       overwrite):
    """Compute source model image with Sherpa.

    Inputs
    ------
    * Source list (JSON file)
    * PSF (JSON file)
    * Exposure image (FITS file)

    Outputs
    -------
    * Source model flux image (FITS file)
    * Source model excess image (FITS file)
    """
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    import sherpa.astro.ui as sau  # @UnresolvedImport
    from ..morphology.psf import Sherpa
    from ..morphology.utils import read_json

    logging.info('Reading exposure: {0}'.format(exposure))
    # Note: We don't really need the exposure as data,
    # but this is a simple way to init the dataspace to the correct shape
    sau.load_data(exposure)
    sau.load_table_model('exposure', exposure)

    logging.info('Reading PSF: {0}'.format(psf))
    Sherpa(psf).set()

    logging.info('Reading sources: {0}'.format(sources))
    read_json(sources, sau.set_source)

    name = sau.get_source().name
    full_model = 'exposure * psf({})'.format(name)
    sau.set_full_model(full_model)

    logging.info('Computing and writing model_image: {0}'.format(model_image))
    sau.save_model(model_image, clobber=overwrite)

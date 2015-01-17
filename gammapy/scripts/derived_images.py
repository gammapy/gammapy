# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['derived_images']


def main(args=None):
    from gammapy.utils.scripts import argparse, GammapyFormatter
    description = derived_images.__doc__.split('\n')[0]
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=GammapyFormatter)
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
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    from astropy.io import fits
    from gammapy.background import Maps

    logging.info('Reading {0}'.format(infile))
    hdus = fits.open(infile)
    maps = Maps(hdus, theta=theta,
                is_off_correlated=is_off_correlated)
    logging.info('Computing derived maps')
    maps.make_derived_maps()
    logging.info('Writing {0}'.format(outfile))
    maps.writeto(outfile, clobber=overwrite)

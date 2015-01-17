# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..utils.scripts import get_parser

__all__ = ['significance_image']


def main(args=None):
    parser = get_parser(significance_image)
    parser.add_argument('infile', type=str,
                        help='Input FITS file name')
    parser.add_argument('outfile', type=str,
                        help='Output FITS file name')
    parser.add_argument('theta', type=float,
                        help='On-region correlation radius (deg)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    significance_image(**vars(args))


def significance_image(infile,
                       outfile,
                       theta,
                       overwrite):
    """Make correlated significance image.

    TODO: describe
    """
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    from astropy.io import fits
    from gammapy.image import disk_correlate
    from gammapy.stats import significance_on_off

    logging.info('Reading {0}'.format(infile))
    hdus = fits.open(infile)
    n_on = hdus['On'].data
    n_off = hdus['Off'].data
    a_on = hdus['OnExposure'].data
    a_off = hdus['OffExposure'].data

    logging.info('Correlating n_on and a_on map')
    theta = theta / hdus['On'].header['CDELT2']
    n_on = disk_correlate(n_on, theta)
    a_on = disk_correlate(a_on, theta)

    logging.info('Computing significance map')
    alpha = a_on / a_off
    significance = significance_on_off(n_on, n_off, alpha)

    logging.info('Writing {0}'.format(outfile))
    fits.writeto(outfile, data=significance, header=hdus['On'].header, clobber=overwrite)

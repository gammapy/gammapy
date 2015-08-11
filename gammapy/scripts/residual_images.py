# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
log = logging.getLogger(__name__)
from ..utils.scripts import get_parser

__all__ = ['residual_images']


def main(args=None):
    parser = get_parser(residual_images)
    parser.add_argument('model_file', type=str,
                        help='Input excess model FITS file name')
    parser.add_argument('data_file', type=str,
                        help='Input data FITS file name')
    parser.add_argument('out_file', type=str,
                        help='Output FITS file name')
    parser.add_argument('--thetas', type=str, default='0.1',
                        help='On-region correlation radii (deg, comma-separated)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    args.thetas = [float(theta) for theta in args.thetas.split(',')]
    residual_images(**vars(args))


def residual_images(model_file,
                    data_file,
                    out_file,
                    thetas,
                    overwrite):
    """Compute source model residual images.

    The input `data_file` must contain the following HDU extensions:

    * 'On' -- Counts image
    * 'Background' -- Background image
    """
    import numpy as np
    from astropy.io import fits
    from gammapy.image import disk_correlate
    from gammapy.stats import significance

    log.info('Reading {0}'.format(data_file))
    hdu_list = fits.open(data_file)
    header = hdu_list[0].header
    counts = hdu_list['On'].data.astype(np.float64)
    background = hdu_list['Background'].data.astype(np.float64)
    diffuse = hdu_list['Diffuse'].data.astype(np.float64)

    log.info('Reading {0}'.format(model_file))
    model = fits.getdata(model_file)

    background_plus_model_diffuse = background + model + diffuse

    out_hdu_list = fits.HDUList()

    for theta in thetas:
        log.info('Processing theta = {0} deg'.format(theta))

        theta_pix = theta / header['CDELT2']
        counts_corr = disk_correlate(counts, theta_pix)
        background_plus_model_corr = disk_correlate(background_plus_model_diffuse, theta_pix)

        # excess_corr = counts_corr - background_plus_model_corr
        significance_corr = significance(counts_corr, background_plus_model_corr)

        name = 'RESIDUAL_SIGNIFICANCE_{0}'.format(theta)
        log.info('Appending HDU extension: {0}'.format(name))
        hdu = fits.ImageHDU(significance_corr, header, name)
        out_hdu_list.append(hdu)

    log.info('Writing {0}'.format(out_file))
    out_hdu_list.writeto(out_file, clobber=overwrite)

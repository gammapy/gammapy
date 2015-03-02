# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from ..utils.scripts import get_parser

__all__ = ['ts_image']


def main(args=None):
    parser = get_parser(ts_image)
    parser.add_argument('input_file', type=str,
                        help='Input data FITS file name')
    parser.add_argument('output_file', type=str,
                        help='Input data FITS file name')
    parser.add_argument('--psf', type=str, default='psf.json',
                        help='JSON file containing PSF information. ')
    parser.add_argument('--morphology', type=str, default='Gaussian2D',
                        help="Which source morphology to use for TS calculation."
                        "Either 'Gaussian2D' or 'Shell2D'.")
    parser.add_argument('--width', type=float, default=None,
                        help="Width of the shell, measured as fraction of the"
                        " inner radius.")
    parser.add_argument('--scales', type=float, default=[0], nargs='+',
                        help='List of scales to compute TS maps for in deg.')
    parser.add_argument('--downsample', type=str, default='auto',
                        help="Downsample factor of the data to obtain optimal"
                        " performance."
                        "Must be power of 2. Can be 'auto' to choose the downsample"
                        "factor automatically depending on the scale.")
    parser.add_argument('--residual', action='store_true',
                        help="Whether to compute a residual TS image. If a residual"
                        "TS image is computed an excess model has to be provided"
                        "using the '--model' parameter.")
    parser.add_argument('--model', type=str,
                        help='Input excess model FITS file name')
    parser.add_argument('--threshold', type=float, default=None,
                        help="Minimal required initial (before fitting) TS value,"
                        " that the fit is done at all.")
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output files.')
    args = parser.parse_args()
    ts_image(**vars(args))


def ts_image(input_file, output_file, psf, model, scales, downsample, residual,
             morphology, width, overwrite, threshold):
    """
    Compute source model residual images.

    The input `data` fits file must contain the following HDU extensions:

    * 'On' -- Counts image
    * 'Background' -- Background image
    * 'Diffuse' -- Diffuse model image
    * 'ExpGammaMap' -- Exposure image
    """
    # Execute script
    import json
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    from astropy.io import fits
    from gammapy.detect import compute_ts_map_multiscale

    # Read data
    logging.info('Reading {0}'.format(input_file))
    maps = fits.open(input_file)
    logging.info('Reading {0}'.format(psf))
    psf_parameters = json.load(open(psf))

    if residual:
        logging.info('Reading {0}'.format(model))
        data = fits.getdata(model)
        header = fits.getheader(model)
        maps.append(fits.ImageHDU(data, header, 'OnModel'))
    results = compute_ts_map_multiscale(maps, psf_parameters, scales, downsample,
                                        residual, morphology, width)

    folder, filename = os.path.split(output_file)
    if not os.path.exists(folder) and folder != '':
        os.mkdir(folder)

    # Write results to file
    header = maps[0].header
    if len(results) > 1:
        for scale, result in zip(scales, results):
            filename_ = filename.replace('.fits', '_{0:.3f}.fits'.format(scale))
            logging.info('Writing {0}'.format(os.path.join(folder, filename_)))
            result.write(os.path.join(folder, filename_), header, overwrite=overwrite)
    else:
        logging.info('Writing {0}'.format(output_file))
        results[0].write(output_file, header, overwrite=overwrite)

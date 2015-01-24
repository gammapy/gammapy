# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from ..utils.scripts import get_parser


def main(args=None):
    parser = get_parser(ts_image)
    parser.add_argument('input', type=str,
                        help='Input data FITS file name')
    parser.add_argument('--folder', type=str, default='gaussian2d',
                        help='Output folder name.')
    parser.add_argument('--psf', type=str, default='psf.json',
                        help='JSON file containing PSF information. ')
    parser.add_argument('--morphology', type=str, default='Gaussian2D',
                        help="Which source morphology to use for TS calculation.\n"
                        "Either 'Gaussian2D' or 'Shell2D'.")
    parser.add_argument('--width', type=float, default=None,
                        help="Width of the shell, measured as fraction of the"
                        " inner radius.\n")
    parser.add_argument('--scales', type=float, default=[0], nargs='+',
                        help='List of scales to compute TS maps for in deg.')
    parser.add_argument('--downsample', type=str, default='auto',
                        help="Downsample factor of the data to obtain optimal"
                        " performance.\n"
                        "Must be power of 2. Can be 'auto' to choose the downsample \n"
                        "factor automatically depending on the scale.")
    parser.add_argument('--residual', action='store_true',
                        help="Whether to compute a residual TS image. If a residual \n"
                        "TS image is computed an excess model has to be provided \n"
                        "using the '--model' parameter.")
    parser.add_argument('--model', type=str,
                        help='Input excess model FITS file name')
    parser.add_argument('--threshold', type=float, default=None,
                        help="Minimal required initial (before fitting) TS value,"
                        " that the \nfit is done at all.")
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output files.')
    args = parser.parse_args()
    ts_image(**vars(args))


def ts_image(input, psf, model, scales, downsample, residual, morphology,
             width, overwrite, folder, threshold):
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
    logging.info('Reading {0}'.format(input))
    maps = fits.open(input)
    logging.info('Reading {0}'.format(psf))
    psf_parameters = json.load(open(psf))

    if residual:
        logging.info('Reading {0}'.format(model))
        data = fits.getdata(model)
        header = fits.getheader(model)
        maps.append(fits.ImageHDU(data, header, 'OnModel'))
    results = compute_ts_map_multiscale(maps, psf_parameters, scales, downsample,
                                        residual, morphology, width)
    folder = morphology.lower()
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Write results to file
    header = maps[0].header
    for scale, result in zip(scales, results):
        filename = os.path.join(folder, 'ts_{0:.3f}.fits'.format(scale))
        logging.info('Writing {0}'.format(filename))
        result.write(filename, header, overwrite=overwrite)


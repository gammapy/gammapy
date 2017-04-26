# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import logging
from ..extern.pathlib import Path
from ..utils.scripts import get_parser, set_up_logging_from_args, make_path
from ..image import SkyImage, SkyImageList
from ..detect import compute_ts_image_multiscale

__all__ = ['run_image_ts']

log = logging.getLogger(__name__)


def image_ts_main(args=None):
    parser = get_parser(run_image_ts)
    parser.add_argument('input_file', type=str,
                        help='Input data FITS file name')
    parser.add_argument('output_file', type=str,
                        help='Output data FITS file name, can contain new or existing folder')
    parser.add_argument('--psf', type=str, default='psf.json',
                        help='JSON file containing PSF information. ')
    parser.add_argument('--morphology', type=str, default='Gaussian2D',
                        help="Which source morphology to use for TS calculation."
                             "Either 'Gaussian2D' or 'Shell2D'.")
    parser.add_argument('--width', type=float, default=None,
                        help="Width of the shell, measured as fraction of the"
                             " inner radius.")
    parser.add_argument('--scales', type=float, default=[0], nargs='+',
                        help='List of scales to compute TS images for in deg.')
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
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output files.')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    parser.add_argument('--threshold', type=float, default=None,
                        help="Minimal required initial (before fitting) TS value,"
                             " that the fit is done at all.")
    args = parser.parse_args(args)
    set_up_logging_from_args(args)
    run_image_ts(**vars(args))


def run_image_ts(input_file, output_file, psf, model, scales, downsample, residual,
                 morphology, width, overwrite, threshold):
    """
    Compute source model residual images.

    The input ``data`` FITS file must contain the following HDU extensions:

    * 'counts' -- Counts image
    * 'background' -- Background image
    * 'exposure' -- Exposure image
    """
    # Read data
    log.info('Reading {}'.format(input_file))
    images = SkyImageList.read(input_file)
    log.info('Reading {}'.format(psf))

    with make_path(psf).open() as fh:
        psf_parameters = json.load(fh)

    if residual:
        log.info('Reading {}'.format(model))
        images['model'] = SkyImage.read(model)

    results = compute_ts_image_multiscale(images, psf_parameters, scales, downsample,
                                          residual, morphology, width, threshold)

    filename = Path(output_file).name
    folder = Path(output_file).parent

    Path(folder).mkdir(exist_ok=True)

    # Write results to file
    if len(results) > 1:
        for scale, result in zip(scales, results):
            # TODO: this is unnecessarily complex
            # Simplify, e.g. by letting the user specify a `base_dir`.
            filename_ = filename.replace('.fits', '_{:.3f}.fits'.format(scale))
            fn = Path(folder) / filename_

            log.info('Writing {}'.format(fn))
            result.write(str(fn), clobber=overwrite)
    elif results:
        log.info('Writing {}'.format(output_file))
        results[0].write(output_file, clobber=overwrite)
    else:
        log.info("No results to write to file: all scales have failed")

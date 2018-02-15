# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import logging
import click
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from ..image import SkyImage, SkyImageList
from ..detect import compute_ts_image_multiscale

log = logging.getLogger(__name__)


@click.command('ts')
@click.argument('input_file', type=str)
@click.argument('output_file', type=str)
@click.option('--psf', default='psf.json',
              help='JSON file containing PSF information. ')
@click.option('--morphology', type=click.Choice(['Gaussian2D', 'Shell2D']), default='Gaussian2D',
              help='Which source morphology to use for TS calculation.')
@click.option('--width', type=float, default=None,
              help='Width of the shell, measured as fraction of the'
                   ' inner radius.')
@click.option('--scales', default='0',
              help='Angular kernel scales in deg (comma-separated list).')
@click.option('--downsample', type=str, default='auto',
              help='Downsample factor of the data to obtain optimal performance.'
                   ' Must be power of 2. Can be "auto" to choose the downsample'
                   ' factor automatically depending on the scale.')
@click.option('--residual', is_flag=True,
              help='Whether to compute a residual TS image. If a residual'
                   ' TS image is computed an excess model has to be provided'
                   ' using the "--model" parameter.')
@click.option('--model', type=str,
              help='Input excess model FITS file name')
@click.option('--threshold', type=float, default=None,
              help='Minimal required initial (before fitting) TS value,'
                   ' that the fit is done at all.')
@click.option('--overwrite', is_flag=True, help='Overwrite output files.')
def cli_image_ts(input_file, output_file, psf, model, scales, downsample, residual,
                 morphology, width, overwrite, threshold):
    """Compute TS image.

    The INPUT_FILE and OUTPUT_FILE arguments are FITS filenames.

    The INPUT_FILE FITS file must contain the following HDU extensions:
    \b
    * 'counts' -- Counts image
    * 'background' -- Background image
    * 'exposure' -- Exposure image
    """
    scales = [float(_) for _ in scales.split(',')]

    # Read data
    log.info('Reading {}'.format(input_file))
    images = SkyImageList.read(input_file)
    log.info('Reading {}'.format(psf))

    with make_path(psf).open() as fh:
        psf_parameters = json.load(fh)

    if residual:
        log.info('Reading {}'.format(model))
        images['model'] = SkyImage.read(model)

    results = compute_ts_image_multiscale(
        images, psf_parameters, scales, downsample,
        residual, morphology, width,
        threshold=threshold,
    )

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
            result.write(str(fn), overwrite=overwrite)
    elif results:
        log.info('Writing {}'.format(output_file))
        results[0].write(output_file, overwrite=overwrite)
    else:
        log.info("No results to write to file: all scales have failed")

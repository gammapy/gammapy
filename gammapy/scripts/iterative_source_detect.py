# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
from ..utils.scripts import get_parser

__all__ = ['iterative_source_detect']

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(iterative_source_detect)
    parser.add_argument('scales', type=float, nargs='*', default=[0.1, 0.2, 0.4],
                        help='List of spatial scales (deg) to search for sources')
    parser.add_argument('--counts', type=str, default='counts.fits',
                        help='Counts FITS file name')
    parser.add_argument('--background', type=str, default='background.fits',
                        help='Background FITS file name')
    parser.add_argument('--exposure', type=str, default='exposure.fits',
                        help='Exposure FITS file name')
    parser.add_argument('output_fits', type=str,
                        help='Output catalog of detections (FITS table format)')
    parser.add_argument('output_regions', type=str,
                        help='Output catalog of detections (ds9 region file format)')
    parser.add_argument('--debug_output_folder', type=str, default='',
                        help='Debug output folder name (empty string for no output)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    iterative_source_detect(**vars(args))


def iterative_source_detect(scales,
                            counts,
                            background,
                            exposure,
                            output_fits,
                            output_regions,
                            debug_output_folder,
                            overwrite):
    """Run an iterative multi-scale source detection.
    """
    from collections import OrderedDict
    import numpy as np
    from astropy.io import fits
    from ..detect import IterativeSourceDetector

    # Load data
    maps = OrderedDict()
    maps['counts'] = counts
    maps['background'] = background
    maps['exposure'] = exposure
    for mapname, filename in maps.items():
        log.info('Reading {0} map: {1}'.format(mapname, filename))
        maps[mapname] = fits.getdata(filename)

    # Compute scales in pixel coordinates
    DEG_PER_PIX = np.abs(fits.getval(counts, 'CDELT1'))
    scales_deg = scales
    scales_pix = np.array(scales_deg) / DEG_PER_PIX
    log.info('Number of scales: {0}'.format(len(scales_deg)))
    log.info('DEG_PER_PIX: {0}'.format(DEG_PER_PIX))
    log.info('Scales in deg: {0}'.format(scales_deg))
    log.info('Scales in pix: {0}'.format(scales_pix))

    # Run the iterative source detection
    detector = IterativeSourceDetector(maps=maps,
                                       scales=scales_pix,
                                       debug_output_folder=debug_output_folder,
                                       overwrite=overwrite)
    detector.run()

    # Save the results
    log.info('Writing {}'.format(output_fits))
    detector.save_fits(output_fits)

    log.info('Writing {}'.format(output_regions))
    detector.save_regions(output_regions)
    # detector.save_json('detect.json')

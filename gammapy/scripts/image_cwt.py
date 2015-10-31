# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import sys
from ..extern.pathlib import Path
from ..utils.scripts import get_parser

__all__ = ['image_cwt']

log = logging.getLogger(__name__)


def image_cwt_main(args=None):
    parser = get_parser(image_cwt)
    parser.add_argument('infile', action="store",
                        help='Input FITS file name')
    parser.add_argument('outfile', action="store",
                        help='Output FITS file name')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    # Wavelet scales to be used
    parser.add_argument('--min_scale', default=6.0, type=float,
                        help='Minimum wavelet scale')
    parser.add_argument('--nscales', default=6, type=int,
                        help='Number of wavelet scales')
    parser.add_argument('--scale_step', default=1.3, type=float,
                        help='Geometric step between wavelet scales')
    # Detection thresholds
    parser.add_argument('--thresh', default=3.0, type=float,
                        help='Significance threshold for pixel detection')
    parser.add_argument('--detect', default=5.0, type=float,
                        help='Significance threshold for source detection')
    parser.add_argument('--niter', default=5, type=int,
                        help='Maximum number of iterations')
    parser.add_argument('--convergence', default=1e-5, type=float,
                        help='Convergence parameter')
    args = parser.parse_args(args)
    image_cwt(**vars(args))


def image_cwt(infile,
              outfile,
              overwrite,
              min_scale,
              nscales,
              scale_step,
              thresh,
              detect,
              niter,
              convergence):
    """Compute filtered image using Continuous Wavelet Transform (CWT).

    TODO: add example and explain output.
    """
    if Path(outfile).is_file() and not overwrite:
        log.error("Output file exists and overwrite is False")
        sys.exit()

    from ..detect.cwt import CWT

    cwt = CWT(min_scale, nscales, scale_step)
    cwt.set_file(infile)
    cwt.iterative_filter_peak(thresh, detect, niter, convergence)
    cwt.save_filter(outfile, overwrite=overwrite)

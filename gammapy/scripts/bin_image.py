# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..utils.scripts import get_parser

__all__ = ['bin_image']


def main(args=None):
    parser = get_parser(bin_image)
    parser.add_argument('event_file', type=str,
                        help='Input FITS event file name')
    parser.add_argument('reference_file', type=str,
                        help='Input FITS reference image file name')
    parser.add_argument('out_file', type=str,
                        help='Output FITS counts cube file name')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    bin_image(**vars(args))


def bin_image(event_file,
              reference_file,
              out_file,
              overwrite):
    """Bin events into an image."""
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    from astropy.io import fits
    from astropy.table import Table
    from gammapy.image.utils import bin_events_in_image

    events = Table.read(event_file)
    reference_image = fits.open(reference_file)[0]
    out_image = bin_events_in_image(events, reference_image)
    out_image.writeto(out_file, clobber=overwrite)

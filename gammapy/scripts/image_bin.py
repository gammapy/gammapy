# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from ..utils.scripts import get_parser
from ..data import EventList
from ..image import SkyImage

__all__ = ['make_counts_image']

log = logging.getLogger(__name__)


def image_bin_main(args=None):
    parser = get_parser(make_counts_image)
    parser.add_argument('event_file', type=str,
                        help='Input FITS event file name')
    parser.add_argument('reference_file', type=str,
                        help='Input FITS reference image file name')
    parser.add_argument('out_file', type=str,
                        help='Output FITS counts cube file name')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    make_counts_image(**vars(args))


def make_counts_image(event_file,
                      reference_file,
                      out_file,
                      overwrite):
    """Bin events into an image."""
    log.info('Reading {}'.format(event_file))
    events = EventList.read(event_file)

    log.info('Reading {}'.format(reference_file))
    image = SkyImage.read(reference_file)

    image.data = np.zeros_like(image.data, dtype='int32')
    image.fill_events(events)

    log.info('Writing {}'.format(out_file))
    image.write(out_file, clobber=overwrite)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click
import numpy as np
from ..data import EventList
from ..image import SkyImage

log = logging.getLogger(__name__)


@click.command('bin')
@click.argument('event_file', type=str)
@click.argument('reference_file', type=str)
@click.argument('out_file', type=str)
@click.option('--overwrite', is_flag=True, help='Overwrite existing files?')
def cli_image_bin(event_file, reference_file, out_file, overwrite):
    """Bin events into an image.

    You have to give the event, reference and out FITS filename.
    """
    log.info('Executing cli_image_bin')

    log.info('Reading {}'.format(event_file))
    events = EventList.read(event_file)

    log.info('Reading {}'.format(reference_file))
    image = SkyImage.read(reference_file)

    image.data = np.zeros_like(image.data, dtype='int32')
    image.fill_events(events)

    log.info('Writing {}'.format(out_file))
    image.write(out_file, overwrite=overwrite)

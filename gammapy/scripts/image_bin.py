# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import click
from gammapy.data import EventList
from gammapy.maps import Map

log = logging.getLogger(__name__)


@click.command("bin")
@click.argument("event_file", type=str)
@click.argument("reference_file", type=str)
@click.argument("out_file", type=str)
@click.option("--overwrite", is_flag=True, help="Overwrite existing files?")
def cli_image_bin(event_file, reference_file, out_file, overwrite):
    """Bin events into an image.

    You have to give the event, reference and out FITS filename.
    """
    log.info("Executing cli_image_bin")

    log.info(f"Reading {event_file}")
    events = EventList.read(event_file)

    log.info(f"Reading {reference_file}")
    m_ref = Map.read(reference_file)

    counts_map = Map.from_geom(m_ref.geom)
    counts_map.fill_events(events)

    log.info(f"Writing {out_file}")
    counts_map.write(out_file, overwrite=overwrite)

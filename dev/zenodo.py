# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility script to work with the .zenodo.json file."""
import json
import logging
import click
from datetime import date

log = logging.getLogger(__name__)


def update_zenodo():
    filename="../.zenodo.json"
    log.info(f"Updating {filename}")
    # add potentially missing content
    with open(filename, "r") as f:
        data = json.load(f)

    data["grants"]["ESCAPE"] = "EU Horizon2020 grant 824064"
    data["grants"]["OSCAR"] = "EU cascading grant 101129751, project 01-385"

    modified_date = str(date.today())
    data["dateModified"] = modified_date

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


@click.command()
@click.option(
    "--log_level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    help="log level",
)
def cli(log_level):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)
    update_zenodo()


if __name__ == "__main__":
    cli()

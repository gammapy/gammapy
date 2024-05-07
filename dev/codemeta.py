# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility script to work with the codemeta.json file."""
import json
import logging
from configparser import ConfigParser
import click

log = logging.getLogger(__name__)


def update_codemeta(maintainer, filename, setup_file=None):
    log.info(f"Updating {filename}")
    # add potentially missing content
    with open(filename, "r") as f:
        data = json.load(f)

    for author in data["author"]:
        if author["familyName"] == maintainer:
            log.info(f"Setting maintainer to {maintainer}")
            data["maintainer"] = author

    data["readme"] = "https://gammapy.org"
    data["issueTracker"] = "https://github.com/gammapy/gammapy/issues"
    data["developmentStatus"] = ("active",)
    data["email"] = "GAMMAPY-COORDINATION-L@IN2P3.FR"

    if setup_file:
        # complete with software requirements from setup.cfg
        cf = ConfigParser()
        cf.read(setup_file)
        requirements = cf["options"]["install_requires"].split("\n")
        if "" in requirements:
            requirements.remove("")
        data["softwareRequirements"] = requirements

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    # replace bad labelled attributes
    with open(filename, "r") as f:
        content = f.read()

    content = content.replace("legalName", "name")
    content = content.replace("version", "softwareVersion")

    with open(filename, "w") as f:
        f.write(content)


@click.command()
@click.option("--maintainer", default="Donath", type=str, help="Maintainer name")
@click.option(
    "--filename", default="../codemeta.json", type=str, help="codemeta filename"
)
@click.option("--setup_file", default="../setup.cfg", type=str, help="setup filename")
@click.option(
    "--log_level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    help="log level",
)
def cli(maintainer, filename, log_level, setup_file):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)
    update_codemeta(maintainer=maintainer, filename=filename, setup_file=setup_file)


if __name__ == "__main__":
    cli()

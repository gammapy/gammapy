import os
import logging
import click
from gammapy.scripts.info import cli_info
from gammapy.scripts.download import cli_download_datasets

log = logging.getLogger(__name__)

RELEASE = "latest"  # Or set to a specific release version if needed


@click.group(name="tutorial")
def cli_tutorial():
    """Gammapy tutorial helper commands."""


@cli_tutorial.command(name="setup")
@click.option(
    "--path",
    default="./gammapy-data",
    show_default=True,
    help="Path to download the datasets if needed.",
)
def cli_tutorial_setup(path):
    """Check tutorial setup and show environment info."""
    if "GAMMAPY_DATA" not in os.environ:
        log.info(f"GAMMAPY_DATA not set. Downloading to '{path}'...")
        cli_download_datasets.callback(out=path, release=RELEASE)
        os.environ["GAMMAPY_DATA"] = path
    else:
        log.info(f"GAMMAPY_DATA is already set to '{os.environ['GAMMAPY_DATA']}'")

    # Call the original info command
    cli_info.callback(system=True, version=True, dependencies=True, envvar=True)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import click
import logging
from gammapy.analysis import Analysis, AnalysisConfig

log = logging.getLogger(__name__)


@click.command(name="config")
def cli_make_config():
    """Writes default configuration file."""
    print("make")
    pass


@click.command(name="run")
def cli_run_analysis():
    """Perform analysis process values declared in configuration file."""
    print("run")
    pass

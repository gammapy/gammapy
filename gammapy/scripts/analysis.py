# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import click
from gammapy.analysis import Analysis, AnalysisConfig

log = logging.getLogger(__name__)


@click.command(name="config")
@click.option(
    "--filename",
    default="config.yaml",
    help="Filename to store the default configuration values.",
    show_default=True,
)
@click.option(
    "--overwrite", default=False, is_flag=True, help="Overwrite existing file."
)
def cli_make_config(filename, overwrite):
    """Writes default configuration file."""
    config = AnalysisConfig()
    config.write(filename, overwrite=overwrite)
    log.info(f"Configuration file produced: {filename}")


@click.command(name="run")
@click.option(
    "--filename",
    default="config.yaml",
    help="Filename with default configuration values.",
    show_default=True,
)
def cli_run_analysis(filename, out, overwrite):
    """Performs automated data reduction process."""
    config = AnalysisConfig.read(filename)
    analysis = Analysis(config)
    analysis.run()

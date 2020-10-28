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
@click.option(
    "--out",
    default="datasets",
    help="Output folder where reduced datasets are stored.",
    show_default=True,
)
@click.option(
    "--overwrite", default=False, is_flag=True, help="Overwrite existing datasets."
)
def cli_run_analysis(filename, out, overwrite):
    """Performs automated data reduction process."""
    config = AnalysisConfig.read(filename)
    config.datasets.background.method = "reflected"
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    analysis.datasets.write(out, overwrite=overwrite)
    log.info(f"Datasets stored in {out} folder.")

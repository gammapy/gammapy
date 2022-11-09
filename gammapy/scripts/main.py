# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import warnings
import click
from gammapy import __version__


# We implement the --version following the example from here:
# http://click.pocoo.org/5/options/#callbacks-and-eager-options
def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    print(f"gammapy version {__version__}")
    ctx.exit()


# http://click.pocoo.org/5/documentation/#help-parameter-customization
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# https://click.palletsprojects.com/en/5.x/python3/#unicode-literals
click.disable_unicode_literals_warning = True


@click.group("gammapy", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--log-level",
    default="info",
    help="Logging verbosity level.",
    type=click.Choice(["debug", "info", "warning", "error"]),
)
@click.option("--ignore-warnings", is_flag=True, help="Ignore warnings?")
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Print version and exit.",
)
def cli(log_level, ignore_warnings):  # noqa: D301
    """Gammapy command line interface (CLI).

    Gammapy is a Python package for gamma-ray astronomy.

    Use ``--help`` to see available sub-commands, as well as the available
    arguments and options for each sub-command.

    For further information, see https://gammapy.org/ and https://docs.gammapy.org/

    \b
    Examples
    --------

    \b
    $ gammapy --help
    $ gammapy --version
    $ gammapy info --help
    $ gammapy info
    """
    logging.basicConfig(level=log_level.upper())

    if ignore_warnings:
        warnings.simplefilter("ignore")


@cli.group("analysis")
def cli_analysis():
    """Automation of configuration driven data reduction process.

    \b
    Examples
    --------

    \b
    $ gammapy analysis config
    $ gammapy analysis run
    $ gammapy analysis config --overwrite
    $ gammapy analysis config --filename myconfig.yaml
    $ gammapy analysis run --filename myconfig.yaml
    """


@cli.group("download", short_help="Download datasets and notebooks")
@click.pass_context
def cli_download(ctx):  # noqa: D301
    """Download notebooks and datasets.

    \b
    Download notebooks published in the Gammapy documentation as well as the
    related datasets needed to execute them.
    \b
    - The option `notebooks` will download the notebook files into a `gammapy-notebooks`
    folder created at the current working directory.
    \b
    - The option `datasets` will download the datasets used in the documentation into a
    `gammapy-datasets` folder created at the current working directory.

    \b
    Examples
    --------

    \b
    $ gammapy download datasets  --out localfolder
    $ gammapy download notebooks --release 0.18 --out localfolder
    """


def add_subcommands():
    from .info import cli_info

    cli.add_command(cli_info)

    from .check import cli_check

    cli.add_command(cli_check)

    from .download import cli_download_notebooks

    cli_download.add_command(cli_download_notebooks)

    from .download import cli_download_datasets

    cli_download.add_command(cli_download_datasets)

    from .analysis import cli_make_config

    cli_analysis.add_command(cli_make_config)

    from .analysis import cli_run_analysis

    cli_analysis.add_command(cli_run_analysis)


add_subcommands()

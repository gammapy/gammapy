# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
import logging
import click
from .. import version


# We implement the --version following the example from here:
# http://click.pocoo.org/5/options/#callbacks-and-eager-options
def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    print('gammapy version {}'.format(version.version))
    ctx.exit()


# http://click.pocoo.org/5/documentation/#help-parameter-customization
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group('gammapy', context_settings=CONTEXT_SETTINGS)
@click.option('--log-level', default='info', help='Logging verbosity level',
              type=click.Choice(['debug', 'info', 'warning', 'error']))
@click.option('--ignore-warnings', is_flag=True, help='Ignore warnings?')
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help='Print version and exit')
def cli(log_level, ignore_warnings):
    """Gammapy command line interface (CLI).

    Gammapy is a Python package for gamma-ray astronomy.

    Use ``--help`` to see available sub-commands, as well as the available
    arguments and options for each sub-command.

    For further information, see http://gammapy.org/ and http://docs.gammapy.org/

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
        warnings.simplefilter('ignore')


@cli.group('image')
def cli_image():
    """Analysis - 2D images"""


def add_subcommands():
    from .info import cli_info
    cli.add_command(cli_info)

    from .check import cli_check
    cli.add_command(cli_check)

    from .image_bin import cli_image_bin
    cli_image.add_command(cli_image_bin)


add_subcommands()

if __name__ == '__main__':
    cli()  # pylint:disable=no-value-for-parameter

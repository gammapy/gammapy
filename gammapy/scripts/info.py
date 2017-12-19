# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging
import importlib
import click
from ..conftest import PYTEST_HEADER_MODULES
from .. import version

log = logging.getLogger(__name__)

GAMMAPY_ENV_VARIABLES = [
    'GAMMAPY_EXTRA',
    'HGPS_DATA',
    'GAMMAPY_FERMI_LAT_DATA',
    'CTADATA',
    'CALDB',
    'GAMMA_CAT',
]


@click.command(name='info')
@click.option('--version/--no-version', default=True, help='Show version info')
@click.option('--dependencies/--no-dependencies', default=True, help='Show dependencies info')
@click.option('--system/--no-system', default=True, help='Show system info')
def cli_info(version, dependencies, system):
    """Display information about Gammapy
    """
    if version:
        info = get_info_version()
        print_info(info=info, title='Gammapy current install')

    if dependencies:
        info = get_info_dependencies()
        print_info(info=info, title='Gammapy dependencies')

    if system:
        info = get_info_system()
        print_info(info=info, title='Gammapy environment variables')


def print_info(info, title):
    """Print Gammapy info."""
    info_all = '\n{title}:\n\n'.format(title=title)

    for key in sorted(info):
        info_all += '\t{key:22s} : {value:<10s} \n'.format(key=key, value=info[key])

    print(info_all)


def get_info_version():
    """Get detailed info about Gammapy version."""
    return {
        'version': version.version,
        'release': str(version.release),
        'githash': version.githash,
    }


def get_info_dependencies():
    """Get info about Gammapy dependencies."""
    info_dependencies = {}
    for label, name in PYTEST_HEADER_MODULES.items():
        try:
            module = importlib.import_module(name)
            module_version = module.__version__
        except ImportError:
            module_version = 'not available'
        info_dependencies[label] = module_version
    return info_dependencies


def get_info_system():
    """Get info about Gammapy environment variables."""
    return {
        name: os.environ.get(name, 'not set')
        for name in GAMMAPY_ENV_VARIABLES
    }

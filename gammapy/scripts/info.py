# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import logging
import importlib
from ..conftest import PYTEST_HEADER_MODULES
from .. import version

GAMMAPY_ENV_VARIABLES = ['GAMMAPY_EXTRA', 'HGPS_DATA', 'GAMMAPY_FERMI_LAT_DATA',
                         'CTADATA', 'CALDB', 'GAMMA_CAT']


log = logging.getLogger(__name__)


def cmd_info(args, parser):
    """Print various info on Gammapy to the console.

    TODO: explain.
    """
    no_args = True

    if args.version or args.all:
        info = get_info_version()
        print_info(info=info, title='Gammapy current install')
        no_args = False

    if args.dependencies or args.all:
        info = get_info_dependencies()
        print_info(info=info, title='Gammapy dependencies')
        no_args = False

    if args.system or args.all:
        info = get_info_system()
        print_info(info=info, title='Gammapy environment variables')
        no_args = False

    if no_args:
        print(parser.description)


def print_info(info, title):
    info_all = '\n{title}:\n\n'.format(title=title)

    for key in sorted(info):
        info_all += '\t{key:22s} : {value:<10s} \n'.format(key=key, value=info[key])

    print(info_all)


def get_info_version():
    info_version = {}
    info_version['version'] = version.version
    info_version['release'] = str(version.release)
    info_version['githash'] = version.githash
    return info_version


def get_info_dependencies():
    """Get info about Gammapy dependencies."""

    info_dependencies = {}
    for label, name in PYTEST_HEADER_MODULES.items():
        try:
            module = importlib.import_module(name)
            version = module.__version__
        except ImportError:
            version = 'not available'
        info_dependencies[label] = version
    return info_dependencies


def get_info_system():
    """Print info about Gammapy dependencies."""
    info_system = {}

    for name in GAMMAPY_ENV_VARIABLES:
        info_system[name] = os.environ.get(name, 'not set')

    return info_system

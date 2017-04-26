# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import logging
import importlib
from ..utils.scripts import get_parser

__all__ = ['print_info']

log = logging.getLogger(__name__)


def print_info_main(args=None):
    parser = get_parser(print_info)
    parser.add_argument('--version', action='store_true',
                        help='Print Gammapy version number')
    # TODO: fix or remove:
    # parser.add_argument('--tools', action='store_true',
    #                     help='Print available command line tools')
    parser.add_argument('--dependencies', action='store_true',
                        help='Print available versions of dependencies')
    args = parser.parse_args(args)

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    print_info(**vars(args))


def print_info(version=False, tools=False, dependencies=False):
    """Print various info on Gammapy to the console.

    TODO: explain.
    """
    if version:
        _print_info_version()

    if tools:
        _print_info_tools()

    if dependencies:
        _print_info_dependencies()


def _print_info_version():
    """Print Gammapy version info."""
    from gammapy import version
    print('\n*** Gammapy version info ***\n')
    print('version: {0}'.format(version.version))
    print('release: {0}'.format(version.release))
    print('githash: {0}'.format(version.githash))
    print('')


def _print_info_tools():
    """Print info about Gammapy command line tools."""
    print('\n*** Gammapy tools ***\n')

    # TODO: how to get a one-line description or
    # full help text from the docstring or ArgumentParser?
    # This is the function names, we want the command-line names
    # that are defined in setup.py !???
    from gammapy.utils.scripts import get_all_main_functions
    scripts = get_all_main_functions()
    names = sorted(scripts.keys())
    for name in names:
        print(name)

    # Old stuff that doesn't work any more ...
    # We assume all tools are installed in the same folder as this script
    # and their names start with "gammapy-".
    # import os
    # from glob import glob
    # DIR = os.path.dirname(__file__)
    # os.chdir(DIR)
    # tools = glob('gammapy-*')
    # for tool in tools:
    #     # Extract first line from docstring as description
    #     description = 'no description available'
    #     lines = open(tool).readlines()
    #     for line in lines:
    #         if line.startswith('"""'):
    #             description = line.strip()[3:]
    #             if description.endswith('"""'):
    #                 description = description[:-3]
    #             break
    #     print('{0:35s} : {1}'.format(tool, description))

    print('')


def _print_info_dependencies():
    """Print info about Gammapy dependencies."""
    print('\n*** Gammapy dependencies ***\n')
    from gammapy.conftest import PYTEST_HEADER_MODULES

    for label, name in PYTEST_HEADER_MODULES.items():
        try:
            module = importlib.import_module(name)
            version = module.__version__
        except ImportError:
            version = 'not available'

        print('{:>20s} -- {}'.format(label, version))

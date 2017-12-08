# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import logging
import importlib
import argparse
import pprint
from ..utils.scripts import get_parser
from ..conftest import PYTEST_HEADER_MODULES
from .. import version


__all__ = ['main']

log = logging.getLogger(__name__)


def main(args):
    """Print various info on Gammapy to the console.

    TODO: explain.
    """
    if args.version or args.all:
        info = get_info_version()
        print_info(info=info, title='Gammapy version')

    if args.dependencies or args.all:
        info = get_info_dependencies()
        print_info(info=info, title='Gammapy dependencies')


def print_info(info, title):
    info_all = '\n*** {title} ***\n\n'.format(title=title)

    for key in sorted(info):
        info_all += '\t{key:12s} : {value:10s} \n'.format(key=key, value=info[key])

    print(info_all)


def get_info_version():
    info_version = {}
    info_version['version'] = version.version
    info_version['release'] = str(version.release)
    info_version['githash'] = version.githash
    return info_version


def get_info_dependencies():
    """Print info about Gammapy dependencies."""

    info_dependencies = {}
    for label, name in PYTEST_HEADER_MODULES.items():
        try:
            module = importlib.import_module(name)
            version = module.__version__
        except ImportError:
            version = 'not available'
        info_dependencies[label] = version
    return info_dependencies

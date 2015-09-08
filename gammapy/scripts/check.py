# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to check various things about a Gammapy installation.

This file is called `check` instead of `test` to prevent confusion
for developers and the test runner from including it in test collection.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import warnings
from ..utils.scripts import get_parser, set_up_logging_from_args

__all__ = [
    'run_tests',
    'run_log_examples',
]

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(run_tests)
    parser.description = 'Check various things about the Gammapy installation'
    subparsers = parser.add_subparsers(help='commands', dest='subparser_name')

    test_parser = subparsers.add_parser('runtests', help='Run tests')
    test_parser.add_argument('--package', type=str, default=None,
                             help='Package to test')

    log_parser = subparsers.add_parser('logging', help='Print logging examples (for debugging)')
    log_parser.add_argument("-l", "--loglevel", default='info',
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            help="Set the logging level")
    args = parser.parse_args(args)
    set_up_logging_from_args(args)

    if args.subparser_name == 'runtests':
        del args.subparser_name
        run_tests(**vars(args))
    elif args.subparser_name == 'logging':
        del args.subparser_name
        run_log_examples(**vars(args))
    else:
        parser.print_help()
        exit(0)


def run_tests(package):
    """Run Gammapy tests."""
    import gammapy
    gammapy.test(package, verbose=True)


def run_log_examples():
    """Run some example code that generates log output.

    This is mainly useful for debugging logging output from Gammapy.
    """
    log.debug('this is log.debug() output')
    log.info('this is log.info() output')
    log.warning('this is log.warning() output')
    warnings.warn('this is warnings.warn() output')

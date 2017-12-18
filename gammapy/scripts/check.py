# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to check various things about a Gammapy installation.

This file is called `check` instead of `test` to prevent confusion
for developers and the test runner from including it in test collection.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import warnings

log = logging.getLogger(__name__)


def cmd_check(args, parser):
    """Check subcommand"""
    parser.print_help()


def cmd_tests(args, parser):
    """Check command to run Gammapy tests."""
    import gammapy
    gammapy.test(args.package, verbose=True)


def cmd_log_examples(args, parser):
    """Check command to run some example code that generates log output."""
    log.debug('this is log.debug() output')
    log.info('this is log.info() output')
    log.warning('this is log.warning() output')
    warnings.warn('this is warnings.warn() output')

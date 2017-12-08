# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to check various things about a Gammapy installation.

This file is called `check` instead of `test` to prevent confusion
for developers and the test runner from including it in test collection.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import warnings
from ..utils.scripts import get_parser, set_up_logging_from_args


log = logging.getLogger(__name__)


def cmd_tests(args, parser):
    """Run Gammapy tests."""
    import gammapy
    gammapy.test(args.package, verbose=True)


def cmd_log_examples(args, parser):
    """Run some example code that generates log output.

    This is mainly useful for debugging logging output from Gammapy.
    """
    log.debug('this is log.debug() output')
    log.info('this is log.info() output')
    log.warning('this is log.warning() output')
    warnings.warn('this is warnings.warn() output')

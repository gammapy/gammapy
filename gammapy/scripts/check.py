# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to check various things about a Gammapy installation.

This file is called `check` instead of `test` to prevent confusion
for developers and the test runner from including it in test collection.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import warnings
import click
from .. import test

log = logging.getLogger(__name__)


@click.group("check")
def cli_check():
    """Run checks for Gammapy"""


@cli_check.command("runtests")
def cli_check_runtests():
    """Run Gammapy tests"""
    test(verbose=True)


@cli_check.command("logging")
def cli_check_logging():
    """Check logging"""
    log.debug("this is log.debug() output")
    log.info("this is log.info() output")
    log.warning("this is log.warning() output")
    warnings.warn("this is warnings.warn() output")

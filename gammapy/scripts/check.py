# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to run tests.

This file is called `check` instead of `test` to prevent confusion
for developers and the test runner from including it in test collection.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..utils.scripts import get_parser

__all__ = ['check']


def main(args=None):
    parser = get_parser(check)
    parser.add_argument('--package', type=str, default=None,
                        help='Package to test')
    args = parser.parse_args(args)
    check(**vars(args))


def check(package):
    """Run gammapy unit tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    import gammapy
    gammapy.test(package, verbose=True)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Top-level script environment for Gammapy.

This is what's executed when you run:

    python -m gammapy

See https://docs.python.org/3/library/__main__.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from .scripts.main import cli

sys.exit(cli())  # pylint:disable=no-value-for-parameter

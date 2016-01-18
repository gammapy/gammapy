# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click

__all__ = ['data_manage']

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
def data_manage():
    """Command line tool for common data management tasks.
    """
    print('Not implemented! Come back later!')

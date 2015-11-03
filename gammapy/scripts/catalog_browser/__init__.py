# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A web app to browse catalogs relevant for gamma-ray astronomy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click

log = logging.getLogger(__name__)

__all__ = []


@click.command()
@click.option('--debug', is_flag=True, help='Run in debug mode?')
@click.option('--port', default=5000, help='Port to run on')
def main(debug, port):
    from .app import create_catalog_browser

    logging.basicConfig(level=logging.INFO)

    # TODO: which config options should be passed to create_app
    # and which set here?
    app = create_catalog_browser(config=None)
    app.run(debug=debug, port=port)

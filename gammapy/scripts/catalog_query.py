# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click
click.disable_unicode_literals_warning = True
from ..catalog import source_catalogs

__all__ = []

log = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Gammapy catalog query command line tool.

    \b
    Examples
    --------
    \b
    gammapy-catalog-query -h
    gammapy-catalog-query catalogs
    gammapy-catalog-query sources 2fhl
    gammapy-catalog-query info 2fhl "2FHL J0534.5+2201"
    gammapy-catalog-query info 3fgl "3FGL J0534.5+2201"
    gammapy-catalog-query info hgps "HESS J1825-137"
    \b
    gammapy-catalog-query table-info 2fhl
    gammapy-catalog-query table-web 2fhl
    """
    pass


@cli.command('catalogs')
def list_catalogs():
    """List available catalogs"""
    source_catalogs.info_table.pprint()


@cli.command('sources')
@click.argument('catalog')
def show_catalog_table(catalog):
    """List sources for CATALOG"""
    catalog = source_catalogs[catalog]
    catalog.table['Source_Name'].pprint(max_lines=-1)


@cli.command('info')
@click.argument('catalog')
@click.argument('source')
def plot_lightcurve(catalog, source):
    """Print info for CATALOG and SOURCE"""
    catalog = source_catalogs[catalog]
    source = catalog[source]

    print()
    print(source)
    print()
    # Generic info dict
    # source.pprint()

    # Specific source info print-out
    # if hasattr(source, 'print_info'):
    #     source.print_info()


@cli.command('table-info')
@click.argument('catalog')
def show_catalog_table(catalog):
    """Summarise table info for CATALOG"""
    catalog = source_catalogs[catalog]
    catalog.table.info()


@cli.command('table-web')
@click.argument('catalog')
def show_catalog_table(catalog):
    """Open table in web browser for CATALOG"""
    catalog = source_catalogs[catalog]
    catalog.table.show_in_browser(jsviewer=True)


@cli.command('plot-lightcurve')
@click.argument('catalog')
@click.argument('source')
def plot_lightcurve(catalog, source):
    """Plot lightcurve for CATALOG and SOURCE"""
    catalog = source_catalogs[catalog]
    source = catalog[source]

    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    ax = source.plot_lightcurve()
    ax.plot()
    plt.tight_layout()
    plt.show()


@cli.command('plot-spectrum')
@click.argument('catalog')
@click.argument('source')
def plot_spectrum(catalog, source):
    """Plot spectrum for CATALOG and SOURCE"""
    catalog = source_catalogs[catalog]
    source = catalog[source]

    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    ax = source.plot_spectrum()
    ax.plot()
    plt.tight_layout()
    plt.show()

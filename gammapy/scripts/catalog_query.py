# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..utils.scripts import get_parser
from ..catalog import source_catalogs

__all__ = ['catalog_query']

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(catalog_query)
    # TODO: get available catalogs from the registry and add
    # an option to print them here.
    parser.add_argument('-c', '--catalog', dest='catalog_name',
                        default='3fgl',
                        choices=['3fgl', '2fhl'],
                        help='Catalog for the source e.g. "3fgl"')
    parser.add_argument('-s', '--source', dest='source_name',
                        help='Source name e.g. "3FGL J0349.9-2102"')
    parser.add_argument('--querytype',
                        choices=['info', 'lightcurve', 'spectrum'],
                        help='The query type: info, lightcurve, or spectrum')
    args = parser.parse_args(args)
    catalog_query(**vars(args))


def catalog_query(catalog_name, source_name, querytype):
    """Query the requested catalog for the requested source.

    Based on the requested querytype return information on the object,
    plot the object's light curve or plot the object's spectrum.
    """
    # TODO: validate inputs and give nice error message instead of traceback?
    catalog = source_catalogs[catalog_name]
    source = catalog[source_name]

    if querytype == 'info':
        print(source.info())
    elif querytype == 'lightcurve':
        import matplotlib.pyplot as plt
        plt.style.use('fivethirtyeight')
        ax = source.plot_lightcurve()
        ax.plot()
        plt.tight_layout()
        plt.show()
    elif querytype == 'spectrum':
        import matplotlib.pyplot as plt
        plt.style.use('fivethirtyeight')
        ax = source.plot_spectrum()
        ax.plot()
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError('Invalid querytype: {}'.format(querytype))

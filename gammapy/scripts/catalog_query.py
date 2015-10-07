# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..utils.scripts import get_parser

__all__ = ['catalog_query']

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(catalog_query)
    parser.add_argument('--catalog',
                        choices=['3FGL'],
                        help='Catalog for the source e.g. 3FGL')
    parser.add_argument('--source',
                        help='Source name e.g. J0349.9-2102')
    parser.add_argument('--querytype',
                        choices=['info', 'lightcurve', 'spectrum'],
                        help='The query type: info, lightcurve, or spectrum')
    args = parser.parse_args(args)
    catalog_query(**vars(args))


def catalog_query(catalog, source, querytype):
    """Query the requested catalog for the requested source.

    Based on the requested querytype return information on the object,
    plot the object's light curve or plot the object's spectrum.
    """
    from gammapy.datasets import fermi

    if catalog == '3FGL':
        catalog_object = fermi.Fermi3FGLObject(source)

    if querytype == 'info':
        print(catalog_object.info())

    elif querytype == 'lightcurve':
        import matplotlib.pyplot as plt

        ax = catalog_object.plot_lightcurve()
        ax.plot()
        plt.show()

    elif querytype == 'spectrum':
        import matplotlib.pyplot as plt

        ax = catalog_object.plot_spectrum()
        ax.plot()
        plt.show()

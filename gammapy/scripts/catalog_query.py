from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..utils.scripts import get_parser

__all__ = ['catalog_query']

def main(args=None):
    parser = get_parser(catalog_query)
    parser.add_argument('catalog',
                        choices=['3FGL'],
                        help='Catalog for the source e.g. 3FGL')
    parser.add_argument('source',
                        help='Source name e.g. J0349.9-2102')
    parser.add_argument('querytype',
                        choices=['info', 'lightcurve', 'spectrum'],
                        help='The query type: info, lightcurve, or spectrum')
    catalog_query(**vars(args))

def catalog_query(catalog, source, querytype):
    """Query a catalog

    TODO: explain.
    """
    from gammapy.datasets import fermi

    if (catalog=='3FGL'):
        catalog_object = fermi.Fermi3FGLObject(source)

    if querytype == 'info':
        #print(source)
        #print ("RA (J2000) " + str(catalog_object.ra))
        #print ("Dec (J2000) " + str(catalog_object.dec))
        #print ("l " + str(catalog_object.glon))
        #print ("b " + str(catalog_object.glat))
        #print ("Flux " + str(catalog_object.int_flux) + " +/- " + str(catalog_object.unc_int_flux)
        #       + " ph /cm2 /MeV /s")
        #print ("Detection significance: " + str(catalog_object.signif) + " sigma")
        raise NotImplementedError

    elif querytype == 'lightcurve':
        raise NotImplementedError

    elif querytype == 'spectrum':
        raise NotImplementedError

    else:
        #print_function("Valid query types are info, light_curve, and spectrum.")
        raise NotImplementedError
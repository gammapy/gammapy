# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
from collections import OrderedDict
import numpy as np
from astropy.extern import six
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.table import vstack as table_vstack
from .utils import skycoord_from_table

__all__ = ['catalog_associate_circle',
           'catalog_combine_associations',
           ]


def catalog_associate_circle(catalog, other_catalog,
                             radius='Association_Radius',
                             other_radius=Angle(0, 'deg')):
    """Find associations within a circle around each source.

    This is convenience function built on `~astropy.coordinates.SkyCoord.search_around_sky`,
    extending it in two ways:

    1. Each source can have a different association radius.
    2. Handle source catalogs (`~astropy.table.Table`) instead of `~astropy.coordinates.SkyCoord`.

    Sources are associated if the sum of their radii is smaller than their separation on the sky.

    Parameters
    ----------
    catalog : `~astropy.table.Table`
        Main source catalog
    other_catalog : `~astropy.table.Table`
        Other source catalog of potential associations
    radius, other_radius : `~astropy.coordinates.Angle` or `str`
        Main source catalog association radius.
        For `str` this must be a column name (in `deg` if without units)

    Returns
    -------
    associations : `~astropy.table.Table`
        The list of associations.
    """
    if isinstance(radius, six.text_type):
        radius = catalog[radius]

    if isinstance(other_radius, six.text_type):
        other_radius = other_catalog[other_radius]

    skycoord = skycoord_from_table(catalog)
    other_skycoord = skycoord_from_table(other_catalog)

    association_catalog_name = getattr(other_catalog, 'name', 'N/A')

    # Compute associations as list of dict and store in `Table` at the end
    associations = []
    for source_index in range(len(catalog)):
        logging.debug('Processing source {} of {}'.format(source_index, len(catalog)))
        # Note: this computation can't be in the inner loop because it's super slow:
        # https://github.com/astropy/astropy/issues/3323#issuecomment-71657245
        # other_catalog['_match_separation'] = source['_match_skycoord'].separation(other_catalog['_match_skycoord'])

        # TODO: check if this is slower or faster than calling `SkyCoord.search_around_sky` here!?

        separation = skycoord[source_index].separation(other_skycoord)
        max_separation = radius[source_index] + other_radius
        other_indices = np.nonzero(separation < max_separation)[0]

        for other_index in other_indices:
            association = OrderedDict(
                Source_Name=catalog['Source_Name'][source_index],
                Association_Catalog=association_catalog_name,
                Association_Name=other_catalog['Source_Name'][other_index],
                Separation=separation[other_index],
            )
            import IPython; IPython.embed(); 1/0
            associations.append(association)

    # Need to define columns if there's not a single association
    names=['Source_Name', 'Association_Catalog', 'Association_Name', 'Separation']
    if len(associations) == 0:
        logging.debug('No associations found.')
        table = Table(names=names)
    else:
        logging.debug('Found {} associations.'.format(len(associations)))
        table = Table(associations, names=names)

    return table


def catalog_combine_associations(associations):
    """Combine (vertical stack) association tables.

    Parameters
    ----------
    associations : dict or (str, `~astropy.table.Table`)
        Associations
    Returns
    -------
    combined_associations : `~astropy.table.Table`
        Combined associations table.
    """
    # Add a column to each table with the catalog name
    for name, table in associations.items():
        logging.debug('{:10s} has {:5d} rows'.format(name, len(table)))
        if len(table) != 0:
            table['Association_Catalog'] = name

    table = table_vstack(list(associations.values()))

    # Sort table columns the way we like it
    names = ['Source_Name', 'Association_Catalog', 'Association_Name', 'Separation']
    table = table[names]

    logging.debug('Combined number of associations: {}'.format(len(table)))

    return table

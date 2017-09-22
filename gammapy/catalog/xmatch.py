# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from collections import OrderedDict
import numpy as np
from ..extern import six
from astropy.coordinates import Angle
from astropy.table import Table, Column
from astropy.table import vstack as table_vstack
from astropy.table import hstack as table_hstack
from astropy.coordinates import SkyCoord
from .utils import skycoord_from_table

__all__ = [
    'catalog_xmatch_circle',
    'catalog_xmatch_combine',
    'table_xmatch_circle_criterion',
    'table_xmatch',
]

log = logging.getLogger(__name__)


def catalog_xmatch_circle(catalog, other_catalog,
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
        For `str` this must be a column name (in degrees if without units)

    Returns
    -------
    associations : `~astropy.table.Table`
        The list of associations.
    """
    if isinstance(radius, six.text_type):
        radius = Angle(catalog[radius])

    if isinstance(other_radius, six.text_type):
        other_radius = Angle(other_catalog[other_radius])

    skycoord = skycoord_from_table(catalog)
    other_skycoord = skycoord_from_table(other_catalog)

    association_catalog_name = other_catalog.meta.get('name', 'N/A')

    # Compute associations as list of dict and store in `Table` at the end
    associations = []
    for source_index in range(len(catalog)):
        # TODO: check if this is slower or faster than calling `SkyCoord.search_around_sky` here!?

        separation = skycoord[source_index].separation(other_skycoord)
        max_separation = radius[source_index] + other_radius
        other_indices = np.nonzero(separation < max_separation)[0]

        for other_index in other_indices:
            association = OrderedDict(
                Source_Index=source_index,
                Source_Name=catalog['Source_Name'][source_index],
                Association_Index=other_index,
                Association_Name=other_catalog['Source_Name'][other_index],
                Association_Catalog=association_catalog_name,
                # There's an issue with scalar `Quantity` objects to init the `Table`
                # https://github.com/astropy/astropy/issues/3378
                # For now I'll just store the values without unit
                Separation=separation[other_index].degree,
            )
            associations.append(association)

    # Need to define columns if there's not a single association
    if len(associations) == 0:
        log.debug('No associations found.')
        table = Table()
        table.add_column(Column([], name='Source_Index', dtype=int))
        table.add_column(Column([], name='Source_Name', dtype=str))
        table.add_column(Column([], name='Association_Index', dtype=int))
        table.add_column(Column([], name='Association_Name', dtype=str))
        table.add_column(Column([], name='Association_Catalog', dtype=str))
        table.add_column(Column([], name='Separation', dtype=float))
    else:
        log.debug('Found {} associations.'.format(len(associations)))
        table = Table(associations, names=associations[0].keys())

    return table


def table_xmatch_circle_criterion(max_separation):
    """An example cross-match criterion for `table_xmatch` that reproduces `catalog_xmatch_circle`.

    TODO: finish implementing this and test it.

    Parameters
    ----------
    max_separation : `~astropy.coordinates.Angle`
        Maximum separation

    Returns
    -------
    xmatch : function
        Cross-match function to be passed to `table_xmatch`.
    """
    def xmatch(row1, row2):
        skycoord1 = SkyCoord(row1['RAJ2000'], row1['DEJ2000'], unit='deg')
        skycoord2 = SkyCoord(row2['RAJ2000'], row2['DEJ2000'], unit='deg')
        separation = skycoord1.separation(skycoord2)
        if separation < max_separation:
            return True
        else:
            return False

    return xmatch


def table_xmatch(table1, table2, xmatch_criterion, return_indices=True):
    """Cross-match rows from two tables with a cross-match criterion callback.

    Note: This is a very flexible and simple way to find matching
    rows from two tables, but it can be very slow, e.g. if you
    create `~astropy.coordinates.SkyCoord` objects or index into them
    in the callback cross-match criterion function:
    https://github.com/astropy/astropy/issues/3323#issuecomment-71657245

    Parameters
    ----------
    table1, table2 : `~astropy.table.Table`
        Input tables
    xmatch_criterion : callable
        Callable that takes two `~astropy.table.Row` objects as input
        and returns True / False when they match / don't match.
    return_indices : bool
        If `True` this function returns a Table with match indices
        ``idx1`` and ``idx2``, if False it stacks the matches in a table using
        `~astropy.table.hstack`.

    Returns
    -------
    matches : `~astropy.table.Table`
        Match table (one match per row)
    """
    matches = Table(names=['idx1', 'idx2'], dtype=[int, int])
    for row1 in table1:
        for row2 in table2:
            if xmatch_criterion(row1, row2):
                matches.add_row([row1.index, row2.index])

    if return_indices:
        return matches
    else:
        raise NotImplementedError
        # TODO: need to sub-set table1 and table1 using the matches
        table = table_hstack([matches, table1, table2])
        return table


def catalog_xmatch_combine(associations):
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
        log.debug('{:10s} has {:5d} rows'.format(name, len(table)))
        if len(table) != 0:
            table['Association_Catalog'] = name

    table = table_vstack(list(associations.values()))

    # Sort table columns the way we like it
    names = ['Source_Name', 'Association_Catalog', 'Association_Name', 'Separation']
    table = table[names]

    log.debug('Combined number of associations: {}'.format(len(table)))

    return table

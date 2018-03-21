# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform basic functions for map and cube analysis.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ..maps import WcsNDMap


def fill_map_counts(event_list, ndmap):
    """Fill a ``WcsMap` with events from an EventList.

     The energy of the events is used for a non-spatial axis homogeneous to energy.
     The other non-spatial axis names should have an entry in the colum names of the ``EventList``

     Parameters
     ----------
     event_list : `~gammapy.data.EventList`
             the input event list
     ndmap : `~gammapy.maps.Map`
         Target map
     """
    # The list will contain the event table entries to be fed into the WcsNDMap
    coord_dict = dict()
    geom = ndmap.geom

    # Add sky coordinates to dictionary
    coord_dict.update(skycoord=event_list.radec)

    # Now check the other axes and find corresponding entries in the EventList
    # energy and time are specific types
    for i, axis in enumerate(geom.axes):
        if axis.type == 'energy':
            # This axis is the energy. We treat it differently because axis.name could be e.g. 'energy_reco'
            coord_dict.update({axis.name: event_list.energy.to(axis.unit)})
        # TODO: add proper extraction for time
        else:
            # We look for other axes name in the table column names (case insensitive)
            try:
                # Here we implicitly assume that there is only one column with the same name
                column_name = next(_ for _ in event_list.table.colnames if _.upper() == axis.name.upper())
                coord_dict.update({axis.name: event_list.table[column_name].to(axis.unit)})
            except StopIteration:
                raise ValueError("Cannot find MapGeom axis {} in EventList", axis.name)
    # Fill it
    ndmap.fill_by_coord(coord_dict)


def make_map_counts(event_list, geom, meta=None):
    """Build a ``WcsNDMap` with events from an EventList.

    The energy of the events is used for a non-spatial axis homogeneous to energy.
    The other non-spatial axis names should have an entry in the colum names of the ``EventList``

    Parameters
    ----------
    event_list : `~gammapy.data.EventList`
            the input event list
    geom : `~gammapy.maps.WcsGeom`
            the reference geometry
    meta : `~collections.OrderedDict`
            Dictionnary of meta information to keep with the map

    Returns
    -------
    cntmap : `~gammapy.maps.WcsNDMap`
        Count cube (3D) in true energy bins
    """
    # Create map
    cntmap = WcsNDMap(geom,meta=meta)
    # Fill it
    fill_map_counts(event_list, cntmap)
    # Add MAPTYPE keyword to identify the nature of the map
    cntmap.meta['MAPTYPE']='COUNTS'

    return cntmap

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform basic functions for map and cube analysis.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ..maps import WcsNDMap

__all__ = [
    'fill_map_counts'
]

def fill_map_counts(count_map, event_list):
    """Fill a ``WcsMap` with events from an EventList.

     The energy of the events is used for a non-spatial axis homogeneous to energy.
     The other non-spatial axis names should have an entry in the colum names of the ``EventList``

     Parameters
     ----------
     count_map : `~gammapy.maps.Map`
         Target map
     event_list : `~gammapy.data.EventList`
             the input event list
     """
    geom = count_map.geom

    # Add sky coordinates to dictionary
    coord_dict=dict(skycoord=event_list.radec)

    # Now check the other axes and find corresponding entries in the EventList
    # energy and time are specific types
    for axis in geom.axes:
        if axis.type == 'energy':
            # This axis is the energy. We treat it differently because axis.name could be e.g. 'energy_reco'
            coord_dict.update({axis.name: event_list.energy.to(axis.unit)})
        # TODO: add proper extraction for time
        else:
            # We look for other axes name in the table column names (case insensitive)
            colnames = [_.upper() for _ in event_list.table.colnames]
            if axis.name.upper() in colnames:
                column_name = event_list.table.colnames[colnames.index(axis.name.upper())]
                coord_dict.update({axis.name: event_list.table[column_name].to(axis.unit)})
            else:
                raise ValueError("Cannot find MapGeom axis {} in EventList", axis.name)
    # Fill it
    count_map.fill_by_coord(coord_dict)
